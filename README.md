# AI-Driven Multi-Intersection Smart Traffic Management System

**Start here:** [`docs/SIMPLE_OVERVIEW.md`](docs/SIMPLE_OVERVIEW.md) explains the whole
project on one page.

In one sentence: each traffic signal decides which road gets green next from how many
vehicles are waiting, and a safety rule above it always clears the way for an ambulance.
Four algorithms are compared (fixed timer, sensor-actuated, max-pressure, and an RL agent),
in SUMO for the real numbers and in a Godot game for the visual demo.

## 0. Visual demo (Godot game + backend)

```bash
pip install matplotlib numpy        # + torch if you want the RL agent in the game
python backend/server.py            # keep this running; dashboard at http://127.0.0.1:8765
```

Then open `game/project.godot` in **Godot 4.3 or newer** and press **Play (F5)**.
Set cars per road, add buses, trucks, bikes, autos or ambulances, pick an algorithm, and
press START. Every run is saved as CSV files plus graphs in `backend/runs/`.
See [`game/README.md`](game/README.md).

## 1. Setup (one time)

Python 3.10–3.12 is recommended.

```bash
python -m venv venv
# Windows: venv\Scripts\activate      Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

`eclipse-sumo` installs SUMO itself, so no separate installer is needed. If you
already have SUMO installed system-wide, make sure `SUMO_HOME` is set.
For a GPU, install the CUDA build of PyTorch from pytorch.org first (optional;
everything here runs on CPU).

Every command below is run **from the project folder**. On Linux/macOS run
`export PYTHONPATH=$(pwd)` first; on Windows `set PYTHONPATH=%cd%`.

## 2. Quick start (pre-trained models included)

The dashboard alone needs only `streamlit`, `pandas` and `plotly`; everything
else is required for training, simulation and detection.


The `models/` and `results/` folders already contain trained models and
benchmark results, so you can go straight to the demo:

```bash
streamlit run dashboard/app.py                         # dashboard in the browser
python run_demo.py --controller max_pressure --seed 42 # watch it live in SUMO-GUI
python run_demo.py --controller ppo_noforecast --seed 42
python run_demo.py --controller fixed --seed 42        # same traffic, fixed timer
```

In SUMO-GUI, emergency vehicles are red; slow the simulation with the *Delay*
box. The terminal prints every signal switch and override with its reason.

## 3. Reproduce everything from scratch

`run_all.sh` (Linux/macOS) or `run_all.bat` (Windows) runs the pipeline.
Step by step commands:

| Step | Command | Output |
|---|---|---|
| Build 2x2 grid network | `python -m sim.build_network` | `sim/data/*.net.xml` |
| Build 3x3 scale network | `python -m sim.build_network --grid-size 3` | `sim/data/*3x3.net.xml` |
| Forecaster training data | `python -m forecast.collect_data --episodes 12` | `results/forecast_series.npy` |
| Train LSTM forecaster | `python -m forecast.train_forecaster --model lstm` | `models/forecaster.pt`, `results/forecast_metrics.json` |
| Train Transformer forecaster | `python -m forecast.train_forecaster --model transformer` | `models/forecaster_transformer.pt` |
| Train MARL (ablation, no forecast) | `python -m rl.train --episodes 100` | `models/ppo_noforecast.pt` |
| Train MARL + forecast (PPO) | `python -m rl.train --episodes 100 --forecast` | `models/ppo_forecast.pt` |
| Train MAPPO + forecast (Centralized Critic) | `python -m rl.train --episodes 100 --algo mappo --forecast` | `models/mappo_forecast.pt` |
| Benchmark all controllers | `python -m evaluate.benchmark` | `results/benchmark_*.csv`, `results/benchmark_table.tex`, `results/logs/*.jsonl` |
| Heavy traffic stress test | `python -m evaluate.benchmark --scale 1.3` | `results/benchmark_*_x1.3.csv` |

Using a real map: export an area from openstreetmap.org (Export → .osm), then
`python -m sim.build_network --osm map.osm` and run the steps above. The RL
observation supports up to 4 phases per junction (`MAX_PHASES` in `config.py`);
complex junctions may need that raised.

## 4. Perception module

```bash
# Lane-wise vehicle counting on an image or video (COCO-pretrained YOLOv8n)
python -m perception.detect_count --source traffic.mp4 --lanes my_lanes.json
# Draw lane regions for your own camera view
python -m perception.draw_lanes --source traffic.mp4 --out my_lanes.json
# Camera -> YOLO -> trained policy -> recommended signal, written to a video
python -m perception.live_camera_control --source traffic.mp4 --lanes my_lanes.json
```

Emergency-vehicle class (needs a labelled dataset, see
`perception/data_template.yaml`; Roboflow Universe and Kaggle have several):

```bash
python -m perception.train_emergency --data my_data.yaml --epochs 50 --device 0
python -m perception.augment_weather --images DS/valid/images --labels DS/valid/labels --out DS/valid_adverse
python -m perception.eval_robustness --model models/emergency.pt --normal my_data.yaml --adverse my_data_adverse.yaml
python -m perception.detect_count --source video.mp4 --emergency-model models/emergency.pt
```

## 5. Project structure

```
config.py                  all paths and tunable constants
sim/build_network.py       SUMO grid (or OSM import), fixed + actuated programs
sim/generate_demand.py     randomised demand: peaks, surges, vehicle mix, EVs, pedestrians
sim/sumo_utils.py          SUMO launch + tripinfo metric parsing
env/intersection.py        phases of one junction + safe signal state machine
env/safety.py              emergency priority, pedestrian min green / max wait
env/traffic_env.py         multi-agent environment, observations, rewards, logging
forecast/                  LSTM & Temporal Transformer: data collection, model, training
rl/ppo.py                  PPO with parameter sharing and action masking
rl/mappo.py                MAPPO with Centralized Critic and Decentralized Actors
rl/controller.py           runs a trained policy + gradient×input explanations
rl/train.py                training loop (supports both PPO and MAPPO)
evaluate/benchmark.py      fair comparison of all controllers + LaTeX export
evaluate/max_pressure.py   max-pressure controller (rule-based, no training)
backend/                   Python backend for the Godot game: algorithms, logging, graphs, dashboard
game/                      Godot 4 game: one junction, four signals, interactive traffic
run_demo.py                live SUMO-GUI demo
perception/                YOLOv8 lane counting, EV fine-tuning, weather augmentation
dashboard/app.py           Streamlit dark-mode glassmorphism dashboard (8 tabs)
dashboard/theme.py         dashboard styling + animated dark theme
.streamlit/config.toml     dashboard dark colour theme
```

## 6. Mapping to the proposal

| Proposal item | Where |
|---|---|
| Vision-based detection per lane, EV class, low-light/weather robustness | `perception/` |
| SUMO multi-intersection network, optional OSM import | `sim/build_network.py` |
| Calibrated demand | `sim/generate_demand.py` (+ `lane_counts.csv` from perception for real calibration) |
| Multi-agent RL (independent PPO, shared observations) | `rl/`, `env/traffic_env.py` |
| Forecasting fed into the RL state | `forecast/`, `forecast` feature in the observation |
| Emergency override + pedestrian minimum | `env/safety.py` |
| Fixed-timer and actuated baselines | `evaluate/benchmark.py` |
| Metrics of Section 8 | `sim/sumo_utils.py`, `evaluate/benchmark.py`, `perception/eval_robustness.py` |
| Interpretable dashboard | `dashboard/app.py`, `rl/controller.py` |

## 7. Results

Five held-out demand scenarios (seeds 42–46), one simulated hour each, identical
traffic for every controller. Mean ± standard deviation. Full data:
`results/benchmark_runs.csv`. Regenerate with `python -m evaluate.benchmark`.

| Controller | Wait (s) | Queue (veh/lane) | Throughput (veh/min) | EV delay (s) | EV stopped (s) | Pedestrian delay (s) | Fuel/veh (g) | CO2 (kg) |
|---|---|---|---|---|---|---|---|---|
| Fixed-time | 80.5 ± 4.5 | 1.31 | 30.8 | 94.1 | 73.2 | 39.4 | 107.9 | 629.5 |
| Actuated (sensor) | 47.2 ± 3.4 | 0.78 | 31.1 | 61.9 | 44.1 | 39.3 | 88.6 | 518.7 |
| **Max-pressure + safety rule** | **42.4 ± 2.5** | **0.69** | 31.1 | 8.1 | 0.0 | 41.2 | **86.6** | **506.9** |
| MARL (no forecast) + safety rule | 52.3 ± 1.3 | 0.85 | 30.9 | 7.3 | 0.0 | 44.5 | 89.3 | 522.2 |
| MARL + forecast + safety rule | 55.7 ± 2.1 | 0.91 | 30.9 | 9.1 | 0.1 | 46.4 | 91.2 | 533.9 |
| MARL + forecast, no safety rule | 56.6 ± 1.2 | 0.93 | 30.9 | 61.3 | 45.6 | 48.4 | 91.5 | 535.5 |

**Findings**

* **Max-pressure + the ambulance/pedestrian safety rule is the best controller**:
  47% less waiting than fixed timing, 10% less than sensor-actuated control, the
  shortest queues and the lowest fuel use and CO2. It needs no training.
* **Emergency vehicles:** delay drops from 62–94 s to 7–9 s under any controller
  that has the safety rule. Without the rule (last row) it is 61 s again, so the
  benefit comes from the rule, not from learning.
* **RL** beats fixed timing (−35%) but not the adaptive rules at 100 training episodes.
* **Forecasting** is accurate on its own (MAE 1.17 vs 1.61 persistence) but did not
  improve control (55.7 s with vs 52.3 s without).
* Throughput is the same for all controllers because demand is below capacity.

**Consistency note:** `config.py` is set to the values the shipped models were
trained with (5 s decision interval, reward = −(halted vehicles + waiting
pedestrians), team weight 0.3). The extra reward terms that were added later are
switched off. With these settings the RL rows reproduce exactly.
MAPPO (`models/mappo_forecast.pt`) was trained for only 1 episode, so it is
experimental and excluded from the default benchmark
(`--controllers mappo_forecast` runs it).

## 8. Troubleshooting

**`ModuleNotFoundError: No module named 'sumolib'` (or torch / cv2) in the dashboard.**
The dashboard reads only the result files and the network XML, so it needs
nothing more than `streamlit`, `pandas` and `plotly`:
`pip install streamlit pandas plotly`. SUMO, PyTorch and YOLO are needed only
for training, simulation and detection — install everything with
`pip install -r requirements.txt`.

**`TypeError: 'str' object cannot be interpreted as an integer` in the dashboard.**
Older Streamlit versions (before 1.46) do not accept `width="stretch"`. The
dashboard now detects the installed version and uses the right argument, so
just make sure you are running the current `dashboard/` folder. Any Streamlit
from 1.28 upward works; `pip install -U streamlit` also fixes it.

**`SUMO_HOME is not set` / `sumo not found`.** Install with
`pip install eclipse-sumo`, or set `SUMO_HOME` to your SUMO installation folder.

**`ModuleNotFoundError: No module named 'config'`.** Run commands from the
project folder with `PYTHONPATH` set to it (see section 1), or use `run_all.sh`
/ `run_all.bat`, which set it for you.

**The dashboard says "No benchmark yet".** Run `python -m evaluate.benchmark`;
it writes `results/benchmark_*.csv` and the replay logs the dashboard reads.

**Training is slow.** One episode is one simulated hour and takes 10–25 s on a
CPU. Use `--episodes 40` for a quick run, or resume a checkpoint with
`python -m rl.train --episodes 40 --start-episode 101 --init models/ppo_forecast.pt --forecast`.

## 9. Honest limitations

* The main network is a synthetic 2×2 grid; OSM import works but was only tested on a small map.
* Demand is synthetic (randomised peaks and surges), not calibrated to measured counts yet.
* Emergency-vehicle *detection* needs a labelled dataset and a GPU training run; until
  then, the simulation reads the emergency flag directly from SUMO (the
  interface the perception module would feed).
* Emergency vehicles are modelled as obeying signals, so the measured benefit is
  purely from signal priority.
* The camera loop uses one camera per junction without tracking, so every
  vehicle in a lane region is treated as queued.
