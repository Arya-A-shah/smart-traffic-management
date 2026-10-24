"""Independent PPO with parameter sharing for multi-agent signal control.

All intersections share one actor-critic network ("parameter sharing"),
but each acts on its own observation, which already contains shared
information from its neighbours (upstream green, team reward). This is
the "independent PPO with shared observations" option from the proposal
and scales to any number of intersections without retraining.

Invalid phases (padding, or the current phase after MAX_GREEN) are
removed with an action mask before sampling.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from config import MAX_PHASES, OBS_DIM


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=MAX_PHASES, hidden=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs, mask):
        h = self.body(obs)
        logits = self.pi(h).masked_fill(~mask, -1e9)
        return logits, self.v(h).squeeze(-1)


class PPO:
    def __init__(self, lr=1e-3, gamma=0.95, lam=0.95, clip=0.2, epochs=10,
                 batch=64, ent_coef=0.01, vf_coef=0.5, hidden=128):
        self.net = ActorCritic(hidden=hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.batch = epochs, batch
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.hidden = hidden

    @torch.no_grad()
    def act(self, obs, masks, deterministic=False):
        o = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        m = torch.as_tensor(np.asarray(masks), dtype=torch.bool)
        logits, v = self.net(o, m)
        dist = Categorical(logits=logits)
        a = logits.argmax(-1) if deterministic else dist.sample()
        return a.numpy(), dist.log_prob(a).numpy(), v.numpy(), dist.probs.numpy()

    def compute_gae(self, traj, last_value):
        """traj: list of dicts (obs, mask, action, logp, value, reward) for ONE agent."""
        adv = np.zeros(len(traj), dtype=np.float32)
        gae, next_v = 0.0, last_value
        for i in reversed(range(len(traj))):
            delta = traj[i]["reward"] + self.gamma * next_v - traj[i]["value"]
            gae = delta + self.gamma * self.lam * gae
            adv[i] = gae
            next_v = traj[i]["value"]
        ret = adv + np.array([t["value"] for t in traj], dtype=np.float32)
        return adv, ret

    def update(self, samples):
        obs = torch.as_tensor(np.stack([s["obs"] for s in samples]), dtype=torch.float32)
        mask = torch.as_tensor(np.stack([s["mask"] for s in samples]), dtype=torch.bool)
        act = torch.as_tensor([s["action"] for s in samples])
        old_logp = torch.as_tensor([s["logp"] for s in samples], dtype=torch.float32)
        adv = torch.as_tensor([s["adv"] for s in samples], dtype=torch.float32)
        ret = torch.as_tensor([s["ret"] for s in samples], dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        stats = []
        for _ in range(self.epochs):
            perm = torch.randperm(len(samples))
            for i in range(0, len(perm), self.batch):
                b = perm[i:i + self.batch]
                logits, v = self.net(obs[b], mask[b])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act[b])
                ratio = torch.exp(logp - old_logp[b])
                s1 = ratio * adv[b]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[b]
                pi_loss = -torch.min(s1, s2).mean()
                v_loss = nn.functional.mse_loss(v, ret[b])
                ent = dist.entropy().mean()
                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                stats.append((pi_loss.item(), v_loss.item(), ent.item()))
        return np.mean(stats, axis=0)

    def save(self, path, extra=None):
        torch.save({"state_dict": self.net.state_dict(), "hidden": self.hidden,
                    **(extra or {})}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(hidden=ckpt.get("hidden", 128))
        agent.net.load_state_dict(ckpt["state_dict"])
        agent.net.eval()
        return agent, ckpt
