"""Run a trained policy (PPO or MAPPO) and explain each decision.

Explanation = gradient x input attribution on the chosen action's logit:
    attribution = (grad_{s} logit_{a}) * s
It indicates which observation entries (e.g. "queue on West approach")
pushed the policy toward the phase it picked, and by how much. This is
visualized in the dashboard under the Explainability tab.
"""
import numpy as np
import torch

from config import FEATURE_NAMES, FEATURES_PER_PHASE, MAX_PHASES
from rl.ppo import PPO

PRETTY = {
    "queue": "queue",
    "vehicles": "vehicles on approach",
    "ped_waiting": "waiting pedestrians",
    "forecast": "forecast inflow",
    "upstream_green": "upstream platoon",
    "emergency": "emergency vehicle",
    "is_current": "currently green",
    "exists": "phase available"
}


class RLController:
    """Inference controller for trained PPO or MAPPO traffic signal policies."""

    def __init__(self, model_path, deterministic=True):
        ckpt_probe = torch.load(model_path, map_location="cpu", weights_only=False)
        if ckpt_probe.get("algo") == "mappo":
            from rl.mappo import MAPPO
            self.agent, self.ckpt = MAPPO.load(model_path)
            self.is_mappo = True
        else:
            self.agent, self.ckpt = PPO.load(model_path)
            self.is_mappo = False

        self.use_forecast = self.ckpt.get("use_forecast", False)
        self.deterministic = deterministic

    def decide(self, env, obs, masks, explain=False):
        tls = list(obs)
        o = np.stack([obs[t] for t in tls])
        m = np.stack([masks[t] for t in tls])
        if self.is_mappo:
            a, _, probs = self.agent.act(o, m, deterministic=self.deterministic)
            # Global centralized value if available
            central_obs = np.concatenate([o.reshape(-1)] * (env.num_intersections if hasattr(env, "num_intersections") else 1))[:self.agent.central_dim]
            if len(central_obs) < self.agent.central_dim:
                central_obs = np.pad(central_obs, (0, self.agent.central_dim - len(central_obs)))
            val = float(self.agent.get_value(central_obs.reshape(1, -1))[0])
            v = [val] * len(tls)
        else:
            a, _, v, probs = self.agent.act(o, m, deterministic=self.deterministic)

        actions = {t: int(a[i]) for i, t in enumerate(tls)}
        meta = {}
        if explain:
            for i, t in enumerate(tls):
                meta[t] = self._explain(env, t, o[i], m[i], actions[t], probs[i], v[i])
        return actions, meta

    def _explain(self, env, tl, obs, mask, action, probs, value):
        x = torch.tensor(obs, requires_grad=True, dtype=torch.float32)
        m_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
        x_in = x.unsqueeze(0)

        if self.is_mappo:
            logits = self.agent.actor(x_in, m_t)
        else:
            logits, _ = self.agent.net(x_in, m_t)

        logits[0, action].backward()
        contrib = (x.grad * x).detach().numpy()[:-1].reshape(MAX_PHASES, FEATURES_PER_PHASE)
        labels = env.inters[tl].labels
        items = []
        for k in range(len(labels)):
            for f, name in enumerate(FEATURE_NAMES):
                if name == "exists" or abs(contrib[k, f]) < 1e-4:
                    continue
                items.append((float(contrib[k, f]), f"{PRETTY[name]} {labels[k]}"))

        items.sort(key=lambda z: -abs(z[0]))
        top = items[:3]
        cur = env.inters[tl].current
        verb = "keep" if action == cur else "switch to"
        factors = ", ".join(f"{n} ({'+' if c > 0 else '-'}{abs(c):.2f})" for c, n in top) if top else "balanced state"
        reason = f"RL: {verb} {labels[action]} (p={probs[action]:.2f}). Main factors: {factors}"

        return {
            "reason": reason,
            "value": float(value),
            "probs": {labels[k]: round(float(probs[k]), 3) for k in range(len(labels))},
            "attribution": [[n, round(c, 3)] for c, n in top]
        }
