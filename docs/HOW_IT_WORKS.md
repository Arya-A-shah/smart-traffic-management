# How the system works (and how to defend it)

## The control loop, one second at a time

1. SUMO advances one second.
2. For every intersection the environment reads, per phase: halted vehicles,
   vehicles on the approach, pedestrians waiting, emergency vehicles present,
   and how green the upstream neighbour currently is toward this phase.
3. Every 30 s, the LSTM forecaster predicts arrivals for the next minute.
4. The safety layer is checked **before** the policy. It can force a phase
   (emergency vehicle, pedestrian waiting > 60 s) or hold one (pedestrians
   crossing, minimum 15 s).
5. If an intersection has been green at least its minimum time, and 5 s have
   passed since its last decision, the shared PPO policy picks the next phase.
6. The intersection object inserts pedestrian clearance (2 s) and yellow (3 s)
   before any new green. The policy can never produce an unsafe transition.

## Key design choices

**Action = which approach gets green next.** The policy is acyclic: it can skip
empty approaches. Transition safety is enforced outside the network.

**Reward = −(halted vehicles + waiting pedestrians), every second.** The sum of
"number of halted vehicles" over every second equals the total seconds spent
waiting (each halted car contributes one second per second). So maximising the
reward is the same as minimising total waiting time. A neighbour term
(0.3 × mean neighbour reward) makes each agent care whether its green floods
the next intersection.

**Independent PPO with parameter sharing.** All four intersections use one
network, so each episode gives four times the data, and the same policy can be
applied to a network of any size. Coordination comes from the shared
observations (upstream green) and the team reward. QMIX was the alternative in
the proposal; PPO was chosen because the phase choice is naturally a
categorical policy, it handles action masking cleanly, and it trained stably.

**Action masking.** Phases that do not exist are masked, and after 60 s of green
the current phase is masked so no approach starves.

**Safety as rules, not learning.** A learned policy has no guarantees and
emergency vehicles are rare in training data. Rules are verifiable. The
ablation `ppo_forecast_nosafety` measures exactly what the rules contribute.

**Forecasting.** Input: the last 5 minutes (10 bins) of arrivals per phase,
upstream green share, and time. Output: arrivals in each of the next two
30 s bins. It is compared with persistence and moving-average forecasts. The
ablation `ppo_noforecast` vs `ppo_forecast` tests whether it helps control.

**Fair benchmark.** Every controller runs on the same five unseen demand
scenarios (same route files, same SUMO seed). Training used different seeds.

## Likely viva questions

*Why not a single agent controlling all four signals?* The joint action space
is 4⁴ = 256 and grows exponentially with intersections; per-intersection
agents scale linearly.

*How do you know it isn't overfitting to training traffic?* Evaluation
scenarios are generated from seeds never used in training, with different peak
times, surges and directional splits.

*What if the policy chooses badly for an ambulance?* It never gets the chance:
the safety layer runs first and forces the ambulance's phase through a normal
clearance and yellow.

*Why is actuated control a strong baseline?* It extends green while vehicles
keep arriving (gap-based), which already adapts to demand. Beating it is the
real test; beating fixed timing is easy.

*How does the camera connect to the RL agent?* `perception/live_camera_control.py`
turns per-lane YOLO counts into the same observation vector used in training.
Features a single camera cannot see (upstream signals, forecast) are zero,
which is why the no-forecast policy is used there.

*What is gradient × input?* For the chosen action's score, the gradient with
respect to each input times the input value approximates how much that input
contributed. The dashboard shows the top three.

*Limitations?* See README section 8.

## Max-pressure (added)

`evaluate/max_pressure.py` is a rule-based controller that uses the same
environment and the same safety layer as the RL agent. At each decision it
computes, per phase, the sum over its green movements of
(halted vehicles on the incoming lane − vehicles on the outgoing lane) and picks
the largest. It needs no training and gave the lowest waiting time in the
benchmark (42.4 s vs 47.2 s actuated). It coordinates junctions without
communication: a congested exit road lowers the pressure of every movement
feeding it.
