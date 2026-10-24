"""Live demo: watch a controller run in the SUMO GUI.

    python run_demo.py --controller ppo_forecast --seed 42
    python run_demo.py --controller fixed --seed 42
    python run_demo.py --controller ppo_forecast --no-gui   # console only

Every RL decision and every safety override is printed with its reason.
Emergency vehicles are red; press the GUI's "Delay" slider to slow down.
"""
import argparse

from config import LOGS, RESULTS
from evaluate.runner import CONTROLLERS, run_controller


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="ppo_forecast", choices=list(CONTROLLERS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--no-gui", action="store_true")
    a = ap.parse_args()

    last_override = {}

    def on_decision(env, actions, meta):
        for tl, it in env.inters.items():
            if it.last_source in ("emergency", "pedestrian") and last_override.get(tl) != it.last_reason:
                print(f"t={env.t:5d}  {tl}: {it.last_reason}")
                last_override[tl] = it.last_reason
        for tl, m in meta.items():
            if actions[tl] != env.inters[tl].current:        # print switches only
                print(f"t={env.t:5d}  {tl}: {m['reason']}")

    stats = run_controller(a.controller, a.seed, RESULTS / "_demo_tripinfo.xml",
                           log_path=LOGS / f"demo_{a.controller}.jsonl",
                           gui=not a.no_gui, on_decision=on_decision, scale=a.scale)
    print("\nEpisode summary:")
    for k, v in stats.items():
        print(f"  {k:28s} {v:.2f}" if isinstance(v, float) else f"  {k:28s} {v}")


if __name__ == "__main__":
    main()
