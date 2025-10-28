"""
step5_pmpo_cartpole.py
PMPO for reinforcement learning using CartPole-v1 environment.
Learn from positive and negative feedback (episode returns).
"""

import gymnasium as gym
import torch, torch.nn as nn, torch.optim as optim
import copy, numpy as np, math

# ---------- Policy network ----------
class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim)
        )
    def forward(self, x):
        return self.net(x)
    def sample(self, obs):
        logits = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        return a, dist.log_prob(a)

def collect_episodes(env, policy, n_episodes=20, max_steps=500):
    """Roll out policy to collect episodes with returns."""
    dataset = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        traj = []
        total_r = 0
        for _ in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            a, logp = policy.sample(obs_t)
            obs_next, r, term, trunc, _ = env.step(a.item())
            traj.append((obs_t.squeeze(0), a, logp))
            total_r += r
            obs = obs_next
            if term or trunc:
                break
        dataset.append({"traj": traj, "return": total_r})
    return dataset

def flatten_feedback(dataset, mask_fn):
    obs, acts, logps = [], [], []
    for ep in dataset:
        if mask_fn(ep["return"]):
            for (o, a, lp) in ep["traj"]:
                obs.append(o); acts.append(a); logps.append(lp)
    return torch.stack(obs), torch.stack(acts), torch.stack(logps)

def kl_divergence(pi_ref, pi_theta, obs_batch):
    with torch.no_grad():
        logits_ref = pi_ref(obs_batch)
    logits_theta = pi_theta(obs_batch)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    probs_theta = torch.softmax(logits_theta, dim=-1)
    kl = (probs_ref * (torch.log(probs_ref + 1e-8) - torch.log(probs_theta + 1e-8))).sum(-1)
    return kl.mean()

# ---------- PMPO training ----------
def train_pmpo_cartpole(alpha=0.6, beta=0.1, steps=100, n_episodes=20):
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = Policy(obs_dim, act_dim)
    pi_ref = copy.deepcopy(policy).eval()
    for p in pi_ref.parameters(): p.requires_grad = False

    opt = optim.Adam(policy.parameters(), lr=3e-3)
    print("Starting PMPO training on CartPole...")

    for step in range(1, steps+1):
        # 1) Collect rollouts
        data = collect_episodes(env, policy, n_episodes=n_episodes)
        returns = np.array([ep["return"] for ep in data])
        thr_hi, thr_lo = np.quantile(returns, 0.75), np.quantile(returns, 0.25)

        pos_obs, pos_acts, _ = flatten_feedback(data, lambda R: R > thr_hi)
        neg_obs, neg_acts, _ = flatten_feedback(data, lambda R: R < thr_lo)

        if len(pos_obs) < 5 or len(neg_obs) < 5: continue

        # 2) Compute log probs for both sets
        logits = policy(pos_obs)
        logp_pos = torch.log_softmax(logits, dim=-1).gather(1, pos_acts.unsqueeze(1)).squeeze(1)

        logits_neg = policy(neg_obs)
        logp_neg = torch.log_softmax(logits_neg, dim=-1).gather(1, neg_acts.unsqueeze(1)).squeeze(1)

        # 3) PMPO mixed objective
        J = alpha * logp_pos.mean() - (1 - alpha) * logp_neg.mean()
        kl = kl_divergence(pi_ref, policy, torch.cat([pos_obs, neg_obs], dim=0))
        loss = -(J - beta * kl)

        opt.zero_grad()
        loss.backward()
        opt.step()

        avg_return = np.mean(returns)
        print(f"[{step:03d}] avg_return={avg_return:.2f} loss={loss.item():.4f}")

        # periodically update reference
        if step % 20 == 0:
            pi_ref = copy.deepcopy(policy).eval()

    print("Training complete.")
    env.close()

if __name__ == "__main__":
    train_pmpo_cartpole()
