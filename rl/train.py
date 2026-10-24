"""Train multi-agent signal controllers (MAPPO or Independent PPO).

Usage:
    python -m rl.train --episodes 60 --forecast               # PPO + forecasting
    python -m rl.train --episodes 60 --algo mappo --forecast  # MAPPO (centralized critic) + forecasting
    python -m rl.train --episodes 60                          # Ablation (no forecast)

Every episode uses a stochastic demand scenario. Every --eval-every
episodes the greedy policy is evaluated on held-out validation scenarios,
and the optimal checkpoint is preserved.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from config import (CENTRAL_OBS_DIM, FORECAST_MODEL, MODELS, OBS_DIM, RESULTS,
                    TRAIN_SEEDS)
from env.traffic_env import TrafficEnv
from forecast.model import Forecaster
from rl.mappo import MAPPO
from rl.ppo import PPO

VAL_SEEDS = [900, 901]


def build_central_obs(obs_dict, tl_ids, target_dim=CENTRAL_OBS_DIM):
    """Concatenate observations across all intersections to form the global state."""
    pieces = [obs_dict[t] for t in tl_ids if t in obs_dict]
    if not pieces:
        return np.zeros(target_dim, dtype=np.float32)
    flat = np.concatenate(pieces, dtype=np.float32)
    if len(flat) < target_dim:
        flat = np.pad(flat, (0, target_dim - len(flat)))
    return flat[:target_dim]


def run_episode(env, agent, seed, deterministic=False, tripinfo=None):
    obs, rew, done, info = env.reset(seed, tripinfo=tripinfo)
    trajs = {tl: [] for tl in env.tl_ids}
    pending = {}
    total_r = 0.0
    is_mappo = isinstance(agent, MAPPO)

    while True:
        # Credit rewards to each agent's previous action
        for tl, r in rew.items():
            if tl in pending:
                pending[tl]["reward"] = r
                trajs[tl].append(pending.pop(tl))
                total_r += r
        if done:
            break

        tls = list(obs)
        o = np.stack([obs[t] for t in tls])
        m = np.stack([info["masks"][t] for t in tls])

        if is_mappo:
            c_obs = build_central_obs(obs, env.tl_ids, agent.central_dim)
            c_val = float(agent.get_value(c_obs.reshape(1, -1))[0])
            a, logp, _ = agent.act(o, m, deterministic)
            for i, tl in enumerate(tls):
                pending[tl] = {
                    "obs": o[i],
                    "mask": m[i],
                    "central_obs": c_obs,
                    "action": int(a[i]),
                    "logp": float(logp[i]),
                    "value": c_val
                }
        else:
            a, logp, v, _ = agent.act(o, m, deterministic)
            for i, tl in enumerate(tls):
                pending[tl] = {
                    "obs": o[i],
                    "mask": m[i],
                    "action": int(a[i]),
                    "logp": float(logp[i]),
                    "value": float(v[i])
                }

        obs, rew, done, info = env.step({tl: int(a[i]) for i, tl in enumerate(tls)})

    # Bootstrap value from final observation
    last_v = {}
    if obs:
        tls = list(obs)
        if is_mappo:
            c_obs = build_central_obs(obs, env.tl_ids, agent.central_dim)
            final_v = float(agent.get_value(c_obs.reshape(1, -1))[0])
            last_v = {tl: final_v for tl in tls}
        else:
            _, _, v, _ = agent.act(np.stack([obs[t] for t in tls]),
                                   np.stack([info["masks"][t] for t in tls]), True)
            last_v = {tl: float(v[i]) for i, tl in enumerate(tls)}

    stats = env.close()
    stats["total_reward"] = total_r
    return trajs, last_v, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--algo", choices=["ppo", "mappo"], default="ppo",
                    help="RL algorithm: independent ppo or mappo with centralized critic")
    ap.add_argument("--forecast", action="store_true", help="use the LSTM/Transformer forecaster")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="continue training from checkpoint")
    ap.add_argument("--start-episode", type=int, default=1, help="episode counter offset when resuming")
    a = ap.parse_args()

    default_name = f"{a.algo}_{'forecast' if a.forecast else 'noforecast'}"
    name = a.name or default_name

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    forecaster = Forecaster(FORECAST_MODEL) if a.forecast else None
    env = TrafficEnv(mode="rl", forecaster=forecaster)

    if a.algo == "mappo":
        agent = MAPPO(obs_dim=OBS_DIM, central_dim=CENTRAL_OBS_DIM)
    else:
        agent = PPO(obs_dim=OBS_DIM)

    log, best = [], float("inf")
    if a.init:
        if a.algo == "mappo":
            loaded, ckpt = MAPPO.load(a.init)
            agent.actor.load_state_dict(loaded.actor.state_dict())
            agent.critic.load_state_dict(loaded.critic.state_dict())
        else:
            loaded, ckpt = PPO.load(a.init)
            agent.net.load_state_dict(loaded.net.state_dict())
        agent.actor.train() if hasattr(agent, "actor") else agent.net.train()
        best = ckpt.get("val_waiting_time_s", best)
        prev = RESULTS / f"training_{name}.json"
        if prev.exists():
            log = [r for r in json.loads(prev.read_text()) if r["episode"] < a.start_episode]

    tripinfo = RESULTS / f"_train_{name}.xml"
    print(f"Starting training run: {name} (Algorithm: {a.algo.upper()}, Forecast: {a.forecast})", flush=True)

    for ep in range(a.start_episode, a.start_episode + a.episodes):
        t0 = time.time()
        seed = TRAIN_SEEDS[(ep * 7 + a.seed) % len(TRAIN_SEEDS)]
        trajs, last_v, st = run_episode(env, agent, seed, tripinfo=tripinfo)

        samples = []
        for tl, traj in trajs.items():
            if not traj:
                continue
            adv, ret = agent.compute_gae(traj, last_v.get(tl, 0.0))
            for s, ad, rt in zip(traj, adv, ret):
                s["adv"], s["ret"] = ad, rt
                samples.append(s)

        pi_l, v_l, ent = agent.update(samples)
        row = {
            "episode": ep,
            "seed": seed,
            "reward": st["total_reward"],
            "avg_waiting_time_s": st["avg_waiting_time_s"],
            "avg_queue_per_lane": st["avg_queue_per_lane"],
            "entropy": float(ent),
            "value_loss": float(v_l),
            "time_s": time.time() - t0
        }
        print(f"[{name}] ep {ep:3d}  reward {row['reward']:8.1f}  wait {row['avg_waiting_time_s']:6.1f}s"
              f"  queue {row['avg_queue_per_lane']:.2f}  ent {ent:.2f}  ({row['time_s']:.0f}s)", flush=True)

        if ep % a.eval_every == 0 or ep == a.start_episode + a.episodes - 1:
            waits = [run_episode(env, agent, s, deterministic=True, tripinfo=tripinfo)[2]
                     ["avg_waiting_time_s"] for s in VAL_SEEDS]
            row["val_waiting_time_s"] = float(np.mean(waits))
            print(f"[{name}]   validation waiting time {row['val_waiting_time_s']:.1f}s", flush=True)
            if row["val_waiting_time_s"] < best:
                best = row["val_waiting_time_s"]
                agent.save(MODELS / f"{name}.pt", {
                    "use_forecast": a.forecast,
                    "episode": ep,
                    "val_waiting_time_s": best,
                    "algo": a.algo
                })
                print(f"[{name}]   --> Saved best checkpoint: models/{name}.pt", flush=True)

        log.append(row)
        (RESULTS / f"training_{name}.json").write_text(json.dumps(log, indent=1))

    tripinfo.unlink(missing_ok=True)
    print(f"Training completed for {name}. Best validation wait time: {best:.1f}s", flush=True)


if __name__ == "__main__":
    main()
