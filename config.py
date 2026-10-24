"""Central configuration for the Smart Traffic Management System.

Every module imports its paths and tunable constants from here, so an
experiment can be configured in one place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SIM_DATA = ROOT / "sim" / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
LOGS = RESULTS / "logs"

for _d in (SIM_DATA, RESULTS, MODELS, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- network
# Two network files with identical geometry. They differ only in the
# default traffic-light program: "static" (fixed timer) or "actuated"
# (SUMO's built-in gap-based actuated controller, our sensor baseline).
NET_STATIC = SIM_DATA / "network.net.xml"
NET_ACTUATED = SIM_DATA / "network_actuated.net.xml"

# Optional 3x3 scale network
NET_STATIC_3X3 = SIM_DATA / "network_3x3.net.xml"
NET_ACTUATED_3X3 = SIM_DATA / "network_actuated_3x3.net.xml"

# ---------------------------------------------------------------- episode
EPISODE_SECONDS = 3600          # one simulated hour per episode
WARMUP_SECONDS = 0

# ---------------------------------------------------------------- signal timing
DELTA = 5            # an agent may re-decide every DELTA seconds of green
                     # (5 s is what the shipped models were trained with)
MIN_GREEN = 10       # minimum green for any phase (s)
MAX_GREEN = 60       # after this, the current phase is masked out (fairness)
CLEARANCE = 2        # pedestrian clearance (vehicle green, crossings red)
YELLOW = 3           # amber time (s)

# ---------------------------------------------------------------- safety layer
EV_DETECTION_RANGE = 200.0   # m from stop line at which an EV triggers priority
PED_MIN_GREEN = 15           # min green when pedestrians are crossing (s)
PED_MAX_WAIT = 60            # a pedestrian never waits longer than this (s)

# ---------------------------------------------------------------- observation
MAX_PHASES = 4               # phases per intersection (padded + masked)
FEATURES_PER_PHASE = 8
OBS_DIM = MAX_PHASES * FEATURES_PER_PHASE + 1
PHASE_CAPACITY = 40.0        # vehicles; used to normalise queue features
FEATURE_NAMES = ["queue", "vehicles", "ped_waiting", "forecast",
                 "upstream_green", "emergency", "is_current", "exists"]

# Centralized critic dimension (concatenation of all 4 intersection observations + global summary)
NUM_INTERSECTIONS = 4
CENTRAL_OBS_DIM = NUM_INTERSECTIONS * OBS_DIM

# ---------------------------------------------------------------- reward engineering
# The reward is deliberately simple:  r = -(halted cars + waiting pedestrians)
# summed over time, that is exactly "minimise total waiting time".
# The extra terms below were experiments added after the shipped models were
# trained; they are switched OFF (0) so code, docs and results agree.
TEAM_REWARD_WEIGHT = 0.3       # r_i + w * mean(r_neighbours)  (coordination)
PED_REWARD_WEIGHT = 1.0        # a waiting pedestrian costs as much as a halted car
THROUGHPUT_REWARD_WEIGHT = 0.0 # experimental, off
EV_PENALTY_WEIGHT = 0.0        # experimental, off (ambulances are handled by the safety rule)
POTENTIAL_SHAPING_WEIGHT = 0.0 # experimental, off
REWARD_SCALE = 0.001

# ---------------------------------------------------------------- forecasting
FORECAST_BIN = 30            # seconds per aggregation bin
FORECAST_HISTORY = 10        # bins of history fed to model (5 minutes)
FORECAST_HORIZON = 2         # bins predicted ahead (1 minute)
FORECAST_MODEL = MODELS / "forecaster.pt"
TRANSFORMER_FORECAST_MODEL = MODELS / "forecaster_transformer.pt"

# Transformer architecture settings
FORECAST_D_MODEL = 64
FORECAST_NHEAD = 4
FORECAST_NUM_LAYERS = 2

# ---------------------------------------------------------------- MAPPO parameters
MAPPO_ACTOR_LR = 3e-4
MAPPO_CRITIC_LR = 5e-4
MAPPO_CLIP = 0.2
MAPPO_EPOCHS = 10
MAPPO_BATCH = 64

# ---------------------------------------------------------------- seeds
TRAIN_SEEDS = list(range(1000, 1200))   # demand scenarios used for training
EVAL_SEEDS = [42, 43, 44, 45, 46]       # held-out scenarios for benchmarking
