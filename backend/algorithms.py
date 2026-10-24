"""The signal-control algorithms used by the Godot game (and the quick comparison).

One junction, 4 approaches, 4 signals. Only one approach is green at a time,
so every phase is conflict-free (cars may go straight, left or right).

    phase 0 = North   1 = East   2 = South   3 = West

Every algorithm answers the same question once per simulated second:
    "Which approach should be green now?"

and every algorithm gets the SAME ambulance-priority rule on top
(it can be switched off to show what the rule is worth).

state (sent by the game every second):
    t             simulation time (s)
    current       approach that is green now
    green_time    how long it has been green (s)
    queue[4]      vehicles stopped / crawling at each red/green line
    approaching[4] all vehicles on each approach road (incl. moving ones)
    detector[4]   vehicles moving inside the 30 m zone just before the stop line
                  (what a road loop sensor sees - used by "actuated")
    ev_dist[4]    distance (m) of the nearest ambulance on each approach, -1 if none
"""
from pathlib import Path

NAMES = ["North", "East", "South", "West"]

MIN_GREEN = 8       # s - a green always lasts at least this long (except for ambulances)
MAX_GREEN = 40      # s - no approach keeps green longer than this (fairness)
FIXED_GREEN = 20    # s - green time per approach for the fixed timer
EV_RANGE = 120      # m - ambulance closer than this triggers priority
SWITCH_MARGIN = 3   # max-pressure: a switch costs ~6 s of yellow/all-red, so only
                    # switch when another approach is clearly ahead

ALGORITHMS = {
    "fixed": {
        "name": "Fixed timer",
        "short": "North -> East -> South -> West, 20 s each, no matter what.",
        "how": "Ignores traffic completely. This is how most old signals work.",
    },
    "actuated": {
        "name": "Actuated (sensor)",
        "short": "Keeps green while cars keep arriving, switches on a gap.",
        "how": ("A loop sensor before the stop line. Green is extended while vehicles keep "
                "crossing it; when nobody arrives for a moment, the next approach WITH cars "
                "gets green. Adapts, but only looks at its own sensor."),
    },
    "max_pressure": {
        "name": "Max-pressure",
        "short": "Gives green to the approach with the most 'pressure' (queue in - queue out).",
        "how": ("Pressure = cars waiting to enter minus cars already on the exit road. "
                "At one junction this means 'serve the longest queue'; in a network the "
                "'minus cars out' part stops it sending cars into a jam - coordination "
                "without any communication. No training needed."),
    },
    "rl": {
        "name": "RL agent (PPO)",
        "short": "A neural network trained in SUMO by trial and error picks the green.",
        "how": ("The trained PPO policy from models/ppo_noforecast.pt. It sees the queues, "
                "vehicle counts, ambulance flags and current green time, and was rewarded "
                "for fewer waiting cars during training."),
    },
}


# --------------------------------------------------------------------- RL
class _RL:
    """Loads the PPO policy trained in SUMO (optional - needs torch)."""
    agent = None
    error = None

    @classmethod
    def load(cls):
        if cls.agent is not None or cls.error is not None:
            return cls.agent
        try:
            import sys
            root = Path(__file__).resolve().parent.parent
            sys.path.insert(0, str(root))
            from rl.ppo import PPO
            cls.agent, _ = PPO.load(root / "models" / "ppo_noforecast.pt")
        except Exception as exc:          # torch missing, model missing, ...
            cls.error = f"{type(exc).__name__}: {exc}"
        return cls.agent


def rl_available():
    return _RL.load() is not None


def _rl_decide(s):
    import numpy as np
    agent = _RL.load()
    obs = np.zeros((4, 8), dtype=np.float32)
    for k in range(4):
        obs[k] = [s["queue"][k] / 40.0, s["approaching"][k] / 40.0, 0.0, 0.0, 0.0,
                  1.0 if s["ev_dist"][k] >= 0 else 0.0,
                  1.0 if k == s["current"] else 0.0, 1.0]
    o = np.concatenate([obs.ravel(), [min(s["green_time"] / 60.0, 1.0)]])[None, :]
    mask = np.ones((1, 4), dtype=bool)
    if s["green_time"] >= MAX_GREEN:
        mask[0, s["current"]] = False
    a, _, _, probs = agent.act(o, mask, deterministic=True)
    a = int(a[0])
    p = probs[0]
    return a, f"network confidence {p[a]*100:.0f}% (N {p[0]:.2f} E {p[1]:.2f} S {p[2]:.2f} W {p[3]:.2f})"


# --------------------------------------------------------------------- rules
def _next_with_demand(s, start):
    """Next approach after `start` (clockwise) that has any vehicles."""
    for i in range(1, 5):
        k = (start + i) % 4
        if s["approaching"][k] > 0:
            return k
    return start


def _algorithm(algo, s):
    cur, g = s["current"], s["green_time"]
    q = s["queue"]

    if algo == "fixed":
        if g >= FIXED_GREEN:
            nxt = (cur + 1) % 4
            return nxt, f"timer: {FIXED_GREEN} s are up, next in the cycle is {NAMES[nxt]}"
        return cur, f"timer: {FIXED_GREEN - g:.0f} s left for {NAMES[cur]}"

    if algo == "actuated":
        if g < MAX_GREEN and s["detector"][cur] > 0:
            return cur, f"sensor still sees {s['detector'][cur]} vehicle(s) arriving on {NAMES[cur]} -> extend green"
        nxt = _next_with_demand(s, cur)
        if nxt == cur:
            return cur, "no vehicles waiting elsewhere -> keep green"
        why = "max green reached" if g >= MAX_GREEN else "gap detected (no arrivals)"
        return nxt, f"{why} -> next approach with traffic: {NAMES[nxt]}"

    if algo == "max_pressure":
        # pressure = vehicles waiting to enter - vehicles on the exit road.
        # The exit roads leave the screen freely, so 'out' is ~0 here.
        pressure = [q[k] + 0.5 * (s["approaching"][k] - q[k]) for k in range(4)]
        if g >= MAX_GREEN:
            pressure[cur] = -1
        best = max(range(4), key=lambda k: (pressure[k], k == cur))
        txt = ", ".join(f"{NAMES[k][0]} {pressure[k]:.1f}" for k in range(4))
        if best == cur:
            return cur, f"{NAMES[cur]} still has the highest pressure ({txt})"
        if pressure[cur] > 0 and pressure[best] - pressure[cur] < SWITCH_MARGIN:
            return cur, f"{NAMES[best]} is only slightly higher - not worth a switch yet ({txt})"
        return best, f"{NAMES[best]} has the highest pressure ({txt})"

    if algo == "rl":
        if not rl_available():
            return _algorithm("max_pressure", s)[0], f"RL unavailable ({_RL.error}); using max-pressure"
        a, why = _rl_decide(s)
        return a, f"RL picked {NAMES[a]}: {why}"

    raise ValueError(f"unknown algorithm {algo}")


def decide(algo, s, ev_priority=True):
    """Returns (phase, reason, source). source in {"emergency", "min_green", "algorithm"}."""
    cur, g = s["current"], s["green_time"]

    # ---- 1. ambulance priority rule (same for every algorithm) ----------
    if ev_priority:
        evs = [(d, k) for k, d in enumerate(s["ev_dist"]) if 0 <= d <= EV_RANGE]
        if evs:
            d, k = min(evs)
            if k == cur:
                return cur, f"AMBULANCE on {NAMES[k]} ({d:.0f} m) -> hold green until it passes", "emergency"
            return k, f"AMBULANCE approaching from {NAMES[k]} ({d:.0f} m) -> switch to green for it", "emergency"

    # ---- 2. minimum green -------------------------------------------------
    if g < MIN_GREEN:
        return cur, f"minimum green ({MIN_GREEN} s) not reached yet", "min_green"

    # ---- 3. the chosen algorithm -------------------------------------------
    k, reason = _algorithm(algo, s)
    return k, reason, "algorithm"
