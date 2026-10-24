# Smart Traffic Management — the simple version

## The project in one sentence

Each traffic signal is an agent that decides **which road gets green next**
from how many vehicles are waiting, and a **safety rule above it always
clears the way for an ambulance**.

## The three pieces

| Piece | What it is | Where |
|---|---|---|
| 1. Simulation | A virtual city where we can test signals safely and fairly. **SUMO** (2×2 grid, 4 junctions) for the real evaluation, and a **Godot game** (1 junction, 4 signals) for the visual demo. | `sim/`, `env/`, `game/` |
| 2. Controller | The algorithm that picks the green. Four are compared (below). | `evaluate/max_pressure.py`, `rl/`, `backend/algorithms.py` |
| 3. Ambulance rule | If an ambulance is within range, its road gets green (after a safe yellow) and keeps it until it has passed. Same rule on top of every controller. | `env/safety.py`, `backend/algorithms.py` |

Everything else (camera detection, forecasting, MAPPO) is **future scope / extras**:
the code is still there, but it is not part of the main story.

## The four algorithms, simplest to smartest

| Algorithm | How it decides | Needs training? | Sees neighbours? |
|---|---|---|---|
| **Fixed timer** | Each road gets a fixed number of seconds, in a cycle | No | No |
| **Actuated** | Keeps green while a road sensor sees cars arriving; switches on a gap | No | No |
| **Max-pressure** | Green to the road with the largest *cars waiting − cars on the exit road*; only switches when clearly worth it | No | Yes, indirectly (a full exit road lowers the pressure) |
| **RL (PPO)** | A neural network trained in SUMO by trial and error; reward = fewer waiting vehicles | Yes (100 simulated hours) | Partly (sees whether the upstream signal is green) |

## Results (SUMO, 2×2 network, 5 unseen one-hour scenarios, same traffic for all)

| Controller | Avg wait (s) | Queue (veh/lane) | Ambulance delay (s) | CO₂ (kg) |
|---|---|---|---|---|
| Fixed timer | 80.5 | 1.31 | 94.1 | 629 |
| Actuated (sensor) | 47.2 | 0.78 | 61.9 | 519 |
| **Max-pressure + ambulance rule** | **42.4** | **0.69** | 8.1 | **507** |
| RL (PPO) + ambulance rule | 52.3 | 0.85 | 7.3 | 522 |
| RL + forecast + ambulance rule | 55.7 | 0.91 | 9.1 | 534 |
| RL + forecast, *no* ambulance rule | 56.6 | 0.93 | 61.3 | 536 |

**What this shows**

1. **Max-pressure + the ambulance rule is the best overall**: 47% less waiting than
   a fixed timer, 10% less than sensor-actuated control, the shortest queues and the
   lowest CO₂, with no training at all.
2. **The ambulance rule is what helps ambulances.** Ambulance delay drops to 7–9 s with
   the rule under any controller; remove it and it goes back up to 61 s.
3. **RL beats the fixed timer but not the simple adaptive rules** at this training
   budget (100 episodes). It is the learning-based approach, with room to improve
   through longer training.
4. **Forecasting did not help control** (55.7 s with vs 52.3 s without), even though
   the forecaster itself is accurate. That is a valid, reportable negative result.

## How to demo it (5 minutes)

1. `python backend/server.py` → open the dashboard at http://127.0.0.1:8765
2. Open `game/project.godot` in Godot 4.3+ and press Play (F5)
3. Choose **Fixed timer**, press START, and add an ambulance on a red road with the
   ambulance rule **off**: it gets stuck. Turn the rule **on**: its road turns green.
4. Switch to **Max-pressure**. The same traffic is replayed, and queues stay shorter.
5. Click **Compare algorithms**: one chart with every algorithm on the same traffic.
6. For the serious numbers, show the SUMO table above (`python -m evaluate.benchmark`).

## Likely viva questions

* *Why is max-pressure "coordinated" if the junctions never talk?* Its pressure
  subtracts the cars already on the exit road. If the next junction is jammed, that
  exit fills up, the pressure drops, and this junction stops sending cars into the jam.
* *Why not just use RL?* RL needs a lot of training and gives no guarantees. A
  well-chosen rule is a strong baseline; RL has to beat it to be worth the cost.
* *Why is the ambulance handled by a rule and not learned?* Ambulances are rare in
  training data, and safety needs guarantees. A rule can be verified and works with
  any controller.
* *Why only one road green at a time in the game?* It keeps every movement
  conflict-free (straight, left and right), so the demo stays easy to read. SUMO uses
  the real multi-movement phases.
