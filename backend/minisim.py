"""A tiny, fast queue simulator used for the "Compare all algorithms" button.

It replays the SAME traffic (same seed, same cars, same ambulances) under every
algorithm in a fraction of a second, using exactly the same decide() function
as the Godot game. It is simpler than the game (cars are points in a queue that
leave one by one at a fixed headway), so treat its numbers as a quick, fair
estimate - the game and SUMO are the detailed versions.
"""
import random

from algorithms import NAMES, decide

TRAVEL_S = 5.0          # seconds from entering the screen to reaching the queue
SPEED_MS = 11.0         # m/s on the approach
YELLOW, ALL_RED = 3, 2
LOST_TIME = 1.5         # start-up delay when a light turns green
HEADWAY = {"car": 2.0, "bike": 1.2, "auto": 1.6, "bus": 3.2, "truck": 3.0, "ambulance": 2.0}
LENGTH_M = {"car": 4.5, "bike": 2.0, "auto": 3.0, "bus": 10.0, "truck": 8.0, "ambulance": 5.0}
MIX = {"car": 0.50, "bike": 0.25, "auto": 0.12, "bus": 0.06, "truck": 0.07}


def make_traffic(scenario):
    """Turn the scenario chosen in the game into a list of vehicles."""
    rng = random.Random(scenario.get("seed", 7))
    dur = scenario.get("duration", 300)
    mix = scenario.get("mix", MIX)
    types, weights = list(mix), list(mix.values())
    out = []
    for k in range(4):
        for i in range(int(scenario.get("initial", [0] * 4)[k])):
            out.append({"t": i * 0.8, "dir": k, "type": rng.choices(types, weights)[0]})
        rate = scenario.get("flows", [0] * 4)[k] / 60.0           # vehicles per second
        t = rng.expovariate(rate) if rate > 0 else dur + 1
        while t < dur:
            out.append({"t": t, "dir": k, "type": rng.choices(types, weights)[0]})
            t += rng.expovariate(rate)
    if scenario.get("random_ambulances"):
        t = rng.uniform(20, 60)
        while t < dur:
            out.append({"t": t, "dir": rng.randrange(4), "type": "ambulance"})
            t += rng.uniform(45, 90)
    for v in scenario.get("extra", []):                       # buttons pressed in the game
        out.append({"t": float(v["t"]), "dir": int(v["dir"]), "type": v["type"]})
    out.sort(key=lambda v: v["t"])
    for i, v in enumerate(out):
        v["id"] = i
    return out


def simulate(algo, scenario, ev_priority=True):
    dur = scenario.get("duration", 300)
    traffic = make_traffic(scenario)
    pending = list(traffic)
    travelling = [[] for _ in range(4)]   # on the road, not yet at the queue
    queue = [[] for _ in range(4)]
    done = []
    current, mode, mode_t, green_t, target = 0, "green", 0.0, 0.0, None
    last_dep = [-99.0] * 4
    series = []
    dt = 0.5
    t = 0.0
    while t < dur or (any(queue) or any(travelling)) and t < dur * 3:
        # arrivals
        while pending and pending[0]["t"] <= t:
            v = pending.pop(0)
            v["arrive_q"] = v["t"] + TRAVEL_S
            travelling[v["dir"]].append(v)
        for k in range(4):
            still = []
            for v in travelling[k]:
                (queue[k] if v["arrive_q"] <= t else still).append(v)
            travelling[k] = still

        # decision once per second
        if abs(t - round(t)) < 1e-9 and mode == "green":
            ev = []
            for k in range(4):
                d = -1
                for i, v in enumerate(queue[k]):
                    if v["type"] == "ambulance":
                        d = sum(LENGTH_M[w["type"]] + 2 for w in queue[k][:i]); break
                if d < 0:
                    for v in travelling[k]:
                        if v["type"] == "ambulance":
                            d = (v["arrive_q"] - t) * SPEED_MS + len(queue[k]) * 6; break
                ev.append(d)
            state = {"t": t, "current": current, "green_time": green_t,
                     "queue": [len(q) for q in queue],
                     "approaching": [len(queue[k]) + len(travelling[k]) for k in range(4)],
                     # loop sensor: cars about to arrive + (on green) cars driving off the queue
                     "detector": [sum(1 for v in travelling[k] if v["arrive_q"] - t < 3)
                                  + (len(queue[k]) if k == current else 0) for k in range(4)],
                     "ev_dist": ev}
            k, _, _ = decide(algo, state, ev_priority)
            if k != current:
                target, mode, mode_t = k, "yellow", 0.0

        # signal state machine
        mode_t += dt
        if mode == "green":
            green_t += dt
        elif mode == "yellow" and mode_t >= YELLOW:
            mode, mode_t = "allred", 0.0
        elif mode == "allred" and mode_t >= ALL_RED:
            current, mode, mode_t, green_t = target, "green", 0.0, 0.0

        # discharge
        if mode == "green" and green_t >= LOST_TIME and queue[current]:
            v = queue[current][0]
            if t - last_dep[current] >= HEADWAY[v["type"]]:
                queue[current].pop(0)
                v["depart"] = t
                v["wait"] = max(0.0, t - v["arrive_q"])
                done.append(v)
                last_dep[current] = t

        if abs(t - round(t)) < 1e-9 and t <= dur:
            series.append([t] + [len(q) for q in queue])
        t += dt

    for k in range(4):                       # vehicles still stuck at the end
        for v in queue[k] + travelling[k]:
            v["wait"] = max(0.0, t - v["arrive_q"])
            done.append(v)
    cars = [v for v in done if v["type"] != "ambulance"]
    amb = [v for v in done if v["type"] == "ambulance"]
    return {
        "algo": algo,
        "vehicles": len(done),
        "avg_wait": sum(v["wait"] for v in cars) / max(len(cars), 1),
        "max_wait": max([v["wait"] for v in cars] or [0]),
        "ambulances": len(amb),
        "amb_wait": sum(v["wait"] for v in amb) / len(amb) if amb else None,
        "max_queue": max(max(r[1:]) for r in series) if series else 0,
        "series": series,
        "by_dir": [sum(v["wait"] for v in cars if v["dir"] == k) / max(1, sum(1 for v in cars if v["dir"] == k))
                   for k in range(4)],
    }


def compare(scenario, algos, ev_priority=True):
    return [simulate(a, scenario, ev_priority) for a in algos]


if __name__ == "__main__":
    sc = {"initial": [8, 3, 10, 2], "flows": [12, 4, 10, 3], "random_ambulances": True,
          "seed": 3, "duration": 300}
    from algorithms import ALGORITHMS
    for r in compare(sc, list(ALGORITHMS)):
        print(f"{r['algo']:14s} wait {r['avg_wait']:6.1f}s  amb {r['amb_wait']}  maxq {r['max_queue']}  n {r['vehicles']}")
    for r in compare(sc, list(ALGORITHMS), ev_priority=False):
        print(f"noEV {r['algo']:14s} wait {r['avg_wait']:6.1f}s  amb {r['amb_wait']}")
    print(NAMES)
