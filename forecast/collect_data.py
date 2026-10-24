"""Record per-intersection inflow time series for forecaster training.

Runs the fixed-time and actuated controllers on many demand scenarios
and stores 30-second inflow bins for every intersection.

Usage: python -m forecast.collect_data --episodes 12
"""
import argparse

import numpy as np

from config import RESULTS, TRAIN_SEEDS
from env.traffic_env import TrafficEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12, help="per controller")
    a = ap.parse_args()
    series = []
    for mode in ("fixed", "actuated"):
        for seed in TRAIN_SEEDS[:a.episodes]:
            env = TrafficEnv(mode=mode, collect_inflow=True)
            env.reset(seed)
            env.run_baseline()
            for tl, s in env.inflow_series.items():
                series.append(np.array(s, dtype=np.float32))
            print(f"{mode} seed {seed}: {len(env.inflow_series)} series")
    np.save(RESULTS / "forecast_series.npy", np.stack(series))
    print("saved", RESULTS / "forecast_series.npy", np.stack(series).shape)


if __name__ == "__main__":
    main()
