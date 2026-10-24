#!/usr/bin/env bash
# Full pipeline, in order. Takes ~1.5-2 hours on a laptop CPU (mostly RL training).
set -e
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)"
cd "$PYTHONPATH"
python -m sim.build_network                        # 1. road network
python -m forecast.collect_data --episodes 12      # 2. traffic data for the forecaster
python -m forecast.train_forecaster --epochs 40    # 3. Forecaster
python -m rl.train --episodes 100                  # 4a. MARL without forecasting (ablation)
python -m rl.train --episodes 100 --forecast       # 4b. MARL with forecasting (proposed PPO)
python -m rl.train --episodes 100 --algo mappo --forecast # 4c. MAPPO with centralized critic
python -m evaluate.benchmark                       # 5. compare against baselines & export LaTeX
echo "Done. Start the dashboard with:  streamlit run dashboard/app.py"
