"""Run one episode with any controller. Shared by benchmark and demo."""
from config import FORECAST_MODEL, MODELS
from env.traffic_env import TrafficEnv

CONTROLLERS = {
    # name: (env mode, model file or None, safety layer on?)
    "fixed": ("fixed", None, False),
    "actuated": ("actuated", None, False),
    "max_pressure": ("rule", "max_pressure", True),
    "ppo_noforecast": ("rl", "ppo_noforecast.pt", True),
    "ppo_forecast": ("rl", "ppo_forecast.pt", True),
    "ppo_forecast_nosafety": ("rl", "ppo_forecast.pt", False),
}
# Experimental: MAPPO was trained for only 1 episode, so it is kept out of the
# default benchmark. Run it explicitly with --controllers mappo_forecast.
EXPERIMENTAL = {
    "mappo_forecast": ("rl", "mappo_forecast.pt", True),
}
CONTROLLERS.update(EXPERIMENTAL)
DEFAULT_BENCHMARK = [c for c in CONTROLLERS if c not in EXPERIMENTAL]
PRETTY = {
    "fixed": "Fixed-time",
    "actuated": "Actuated (sensor)",
    "max_pressure": "Max-pressure (rule-based)",
    "ppo_noforecast": "MARL (no forecast)",
    "ppo_forecast": "MARL + forecast (proposed)",
    "ppo_forecast_nosafety": "MARL + forecast, no safety layer",
    "mappo_forecast": "MAPPO + forecast (experimental)",
}


def run_controller(name, seed, tripinfo, log_path=None, gui=False, explain=False,
                   on_decision=None, scale=1.0):
    mode, model, safety = CONTROLLERS[name]
    if mode == "rule":
        from evaluate.max_pressure import MaxPressureController
        ctrl = MaxPressureController()
        env = TrafficEnv(mode="rl", safety=safety, gui=gui, log_path=log_path)
        obs, _, done, info = env.reset(seed, tripinfo=tripinfo, scale=scale)
        while not done:
            actions, meta = ctrl.decide(env, obs, info["masks"])
            if on_decision:
                on_decision(env, actions, meta)
            obs, _, done, info = env.step(actions, meta)
        return env.close()

    if mode != "rl":
        env = TrafficEnv(mode=mode, gui=gui, log_path=log_path)
        env.reset(seed, tripinfo=tripinfo, scale=scale)
        return env.run_baseline()

    from forecast.model import Forecaster
    from rl.controller import RLController
    ctrl = RLController(MODELS / model)
    forecaster = Forecaster(FORECAST_MODEL) if ctrl.use_forecast else None
    env = TrafficEnv(mode="rl", forecaster=forecaster, safety=safety, gui=gui,
                     log_path=log_path)
    obs, _, done, info = env.reset(seed, tripinfo=tripinfo, scale=scale)
    while not done:
        actions, meta = ctrl.decide(env, obs, info["masks"], explain=explain or gui)
        if on_decision:
            on_decision(env, actions, meta)
        obs, _, done, info = env.step(actions, meta)
    return env.close()
