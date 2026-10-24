"""Multi-Agent PPO (MAPPO) with Centralized Training and Decentralized Execution (CTDE).

In MAPPO:
- Decentralized Actors: Each traffic intersection chooses its action based on
  its own local observation (lane queues, arrivals, local forecasts, neighbor signals).
  Parameter sharing is utilized across all intersection actors.
- Centralized Critic: A unified value network observes the joint global traffic state
  (concatenation of all intersections' observations across the road network).
  This eliminates multi-agent non-stationarity and significantly reduces gradient variance.
- Action Masking: Invalid or unsafe phases (padding, or current phase exceeding MAX_GREEN)
  are strictly masked out before action sampling.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from config import CENTRAL_OBS_DIM, MAX_PHASES, OBS_DIM


class DecentralizedActor(nn.Module):
    """Actor network with LayerNorm for stable policy optimization."""

    def __init__(self, obs_dim=OBS_DIM, n_actions=MAX_PHASES, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, obs, mask):
        logits = self.net(obs)
        return logits.masked_fill(~mask, -1e9)


class CentralizedCritic(nn.Module):
    """Centralized value function estimating global network return V(s_global)."""

    def __init__(self, central_dim=CENTRAL_OBS_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(central_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, central_obs):
        return self.net(central_obs).squeeze(-1)


class MAPPO:
    """MAPPO Controller implementing CTDE for network-level traffic signals."""

    def __init__(self, obs_dim=OBS_DIM, central_dim=CENTRAL_OBS_DIM, n_actions=MAX_PHASES,
                 actor_lr=3e-4, critic_lr=5e-4, gamma=0.95, lam=0.95, clip=0.2,
                 epochs=10, batch=64, ent_coef=0.01, vf_coef=0.5, hidden_actor=128,
                 hidden_critic=256):
        self.actor = DecentralizedActor(obs_dim, n_actions, hidden=hidden_actor)
        self.critic = CentralizedCritic(central_dim, hidden=hidden_critic)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.batch = epochs, batch
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.hidden_actor = hidden_actor
        self.hidden_critic = hidden_critic
        self.obs_dim = obs_dim
        self.central_dim = central_dim

    @torch.no_grad()
    def act(self, obs, masks, deterministic=False):
        """Decentralized inference: actor acts only on local observations and masks."""
        o = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        m = torch.as_tensor(np.asarray(masks), dtype=torch.bool)
        logits = self.actor(o, m)
        dist = Categorical(logits=logits)
        a = logits.argmax(-1) if deterministic else dist.sample()
        return a.numpy(), dist.log_prob(a).numpy(), dist.probs.numpy()

    @torch.no_grad()
    def get_value(self, central_obs):
        """Centralized value estimation from joint global state."""
        c = torch.as_tensor(np.asarray(central_obs), dtype=torch.float32)
        return self.critic(c).numpy()

    def compute_gae(self, traj, last_value):
        """Compute Generalized Advantage Estimation across the centralized trajectory."""
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
        """Update both decentralized actor and centralized critic using collected experience."""
        obs = torch.as_tensor(np.stack([s["obs"] for s in samples]), dtype=torch.float32)
        mask = torch.as_tensor(np.stack([s["mask"] for s in samples]), dtype=torch.bool)
        central_obs = torch.as_tensor(np.stack([s["central_obs"] for s in samples]), dtype=torch.float32)
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

                # Actor update
                logits = self.actor(obs[b], mask[b])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act[b])
                ratio = torch.exp(logp - old_logp[b])
                s1 = ratio * adv[b]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[b]
                pi_loss = -torch.min(s1, s2).mean()
                ent = dist.entropy().mean()
                total_pi_loss = pi_loss - self.ent_coef * ent

                self.actor_opt.zero_grad()
                total_pi_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()

                # Centralized Critic update
                v_pred = self.critic(central_obs[b])
                v_loss = nn.functional.mse_loss(v_pred, ret[b])

                self.critic_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_opt.step()

                stats.append((pi_loss.item(), v_loss.item(), ent.item()))

        return np.mean(stats, axis=0)

    def save(self, path, extra=None):
        torch.save({
            "algo": "mappo",
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "hidden_actor": self.hidden_actor,
            "hidden_critic": self.hidden_critic,
            **(extra or {})
        }, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(
            hidden_actor=ckpt.get("hidden_actor", 128),
            hidden_critic=ckpt.get("hidden_critic", 256)
        )
        agent.actor.load_state_dict(ckpt["actor_state_dict"])
        if "critic_state_dict" in ckpt:
            agent.critic.load_state_dict(ckpt["critic_state_dict"])
        agent.actor.eval()
        agent.critic.eval()
        return agent, ckpt
