"""Benchmark all controllers on held-out demand scenarios.

Every controller sees exactly the same traffic (same route files, same
SUMO seed) so the comparison is rigorous and scientifically sound.

Outputs:
    results/benchmark_runs.csv      one row per (controller, scenario)
    results/benchmark_summary.csv   mean and std per controller
    results/benchmark_table.tex     LaTeX formatted publication table for thesis/report
    results/logs/<controller>.jsonl decision + state log for the dashboard

Usage:
    python -m evaluate.benchmark [--seeds 42 43 44 45 46] [--scale 1.0]
"""
import argparse
from pathlib import Path
import pandas as pd

from config import EVAL_SEEDS, LOGS, MODELS, RESULTS
from evaluate.runner import CONTROLLERS, DEFAULT_BENCHMARK, PRETTY, run_controller

METRICS = [
    "avg_waiting_time_s", "avg_time_loss_s", "avg_queue_per_lane",
    "throughput_veh_per_min", "emergency_time_loss_s", "emergency_waiting_s",
    "ped_time_loss_s", "avg_fuel_g", "total_co2_kg"
]

METRIC_HEADERS = {
    "avg_waiting_time_s": "Wait Time (s)",
    "avg_time_loss_s": "Time Loss (s)",
    "avg_queue_per_lane": "Queue (veh/lane)",
    "throughput_veh_per_min": "Throughput (v/m)",
    "emergency_time_loss_s": "EV Delay (s)",
    "emergency_waiting_s": "EV Stopped (s)",
    "ped_time_loss_s": "Ped Delay (s)",
    "avg_fuel_g": "Fuel/veh (g)",
    "total_co2_kg": "CO2 (kg)"
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=EVAL_SEEDS)
    ap.add_argument("--scale", type=float, default=1.0, help="demand multiplier")
    ap.add_argument("--controllers", nargs="*", default=DEFAULT_BENCHMARK)
    a = ap.parse_args()

    rows = []
    for name in a.controllers:
        mode, model = CONTROLLERS[name][0], CONTROLLERS[name][1]
        if mode == "rl" and model and not (MODELS / model).exists():
            print(f"Skipping {name}: model checkpoint '{model}' not found in models/")
            continue
        for i, seed in enumerate(a.seeds):
            log = LOGS / f"{name}.jsonl" if i == 0 else None
            st = run_controller(
                name, seed, RESULTS / "_bench_tripinfo.xml",
                log_path=log, explain=log is not None, scale=a.scale
            )
            st.update({"controller": name, "seed": seed})
            rows.append(st)
            print(f"{PRETTY[name]:34s} seed {seed:2d}: wait {st['avg_waiting_time_s']:6.1f}s  "
                  f"queue {st['avg_queue_per_lane']:.2f}  EV loss {st['emergency_time_loss_s']:5.1f}s",
                  flush=True)

    (RESULTS / "_bench_tripinfo.xml").unlink(missing_ok=True)

    if not rows:
        print("No controllers evaluated.")
        return

    df = pd.DataFrame(rows)
    suffix = "" if a.scale == 1.0 else f"_x{a.scale:g}"
    runs_file = RESULTS / f"benchmark_runs{suffix}.csv"
    summary_file = RESULTS / f"benchmark_summary{suffix}.csv"
    latex_file = RESULTS / f"benchmark_table{suffix}.tex"

    df.to_csv(runs_file, index=False)

    summ = df.groupby("controller", sort=False)[METRICS].agg(["mean", "std"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ.insert(0, "label", [PRETTY.get(c, c) for c in summ.index])
    summ.to_csv(summary_file)

    # Formatted publication table
    means = df.groupby("controller", sort=False)[METRICS].mean()
    means.index = [PRETTY.get(c, c) for c in means.index]
    means.columns = [METRIC_HEADERS.get(c, c) for c in means.columns]
    print("\n=== BENCHMARK SUMMARY (Averaged across held-out scenarios) ===")
    print(means.round(2).to_string())

    # Export LaTeX table for final report / paper
    try:
        latex_str = means.round(2).to_latex(
            caption="Evaluation of Traffic Control Systems across Held-Out Stochastic Demand Scenarios.",
            label="tab:traffic_benchmarks",
            position="htbp"
        )
        latex_file.write_text(latex_str)
        print(f"\nLaTeX table exported to: {latex_file}")
    except Exception as exc:
        print(f"Could not generate LaTeX export: {exc}")

    # Relative improvement report
    def mean(c, m):
        return df[df.controller == c][m].mean()
    for prop in ("max_pressure", "ppo_noforecast"):
        if prop not in df.controller.values:
            continue
        for base in ("fixed", "actuated"):
            if base not in df.controller.values:
                continue
            bw, pw = mean(base, "avg_waiting_time_s"), mean(prop, "avg_waiting_time_s")
            be, pe = mean(base, "emergency_time_loss_s"), mean(prop, "emergency_time_loss_s")
            print(f"\n[{PRETTY[prop]} vs {PRETTY[base]}]")
            print(f"  Vehicle waiting time: {100 * (bw - pw) / bw:+.1f}% (positive = better)")
            print(f"  Ambulance delay:      {100 * (be - pe) / be:+.1f}% (positive = better)")


if __name__ == "__main__":
    main()
