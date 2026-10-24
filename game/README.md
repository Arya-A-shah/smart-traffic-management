# Smart Traffic Junction (Godot 4 game)

One junction, four signals (North, East, South, West), left-hand traffic.
Only one road is green at a time, so every movement is conflict-free.

## Run it

1. Start the backend (from the project folder):
   ```bash
   pip install matplotlib numpy      # add torch for the RL agent
   python backend/server.py
   ```
2. Install **Godot 4.3+** (standard build, not .NET) from https://godotengine.org/download
3. Open Godot → **Import** → choose `game/project.godot` → **Run (F5)**

If the backend is not running, the game still works but uses a plain fixed timer
(the panel says "Backend: OFFLINE"). Start the backend at any time and it reconnects.

## Controls

| Panel | What it does |
|---|---|
| 1 Algorithm | Fixed timer / Actuated / Max-pressure / RL agent. Switching restarts the **same** traffic so runs compare fairly. Tick or untick the **ambulance priority rule**. |
| 2 Traffic | Per road: vehicles already waiting at the start, new vehicles per minute, and **+1 / +5** buttons to add vehicles now. |
| 3 Special vehicle | Pick a road, then add a Car, Bike, Auto, Bus, Truck or **Ambulance**. Optional random ambulances. |
| 4 Run | START / PAUSE (or Space), Restart, New traffic (new random seed), speed 1x–8x. |
| 5 Results | **Compare algorithms**: the backend replays your settings under every algorithm and shows a chart. **Dashboard**: opens live graphs in the browser. |

The right-hand panel shows which road is green and **why** (the reason comes from
the algorithm), live numbers, a queue chart and the last signal changes.

## What gets saved

Every run goes to `backend/runs/<time>_<n>_<algorithm>/`:
`timeseries.csv` (queues each second), `vehicles.csv` (wait time of every vehicle),
`decisions.csv` (every switch and its reason), `summary.json`, `queues.png` and
`waits_by_type.png`. The last comparison is in `backend/runs/last_compare.png`.

## Launch options (for demos/tests)

```bash
godot --path game -- --algo=2 --speed=4 --autostart --ambulance=1
```
`--algo` 0 fixed, 1 actuated, 2 max-pressure, 3 RL · `--seed=N` · `--ambulance=<road 0-3>`
