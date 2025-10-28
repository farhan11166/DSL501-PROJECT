"""
step1_pmpo_sphere.py
Simple PMPO (positive-only) on Sphere synthetic benchmark.
Implements Section 3.1 weighted-MLE update using samples from a reference policy.

Run:
    python step1_pmpo_sphere.py
"""

import math
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# Config / Hyperparameters
# ----------------------------
DEVICE = "cpu"                # change to "cuda" if you have GPU
DIM = 10                      # dimension of action space
BATCH = 512                   # number of samples per iteration (from pi_ref)
STEPS = 2000                  # training iterations
ETA = 0.5                     # temperature for positive weighting
LR = 3e-3
EVAL_SAMPLES = 2048
SEED = 0

# ----------------------------
# Utilities / Seed
# ----------------------------
def set_seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

set_seed(SEED)

# ----------------------------
# Sphere function (minimize)
# ----------------------------
def sphere(x: torch.Tensor) -> torch.Tensor:
    # x: (B, DIM)
    return torch.sum(x * x, dim=-1)

def reward_from_x(x: torch.Tensor) -> torch.Tensor:
    # r = -f(x) so larger r = better
    return -sphere(x)

# ----------------------------
# Diagonal Gaussian Policy
# ----------------------------
class DiagGaussianPolicy(nn.Module):
    def __init__(self, dim: int, hidden: int = 64, min_log_std=-5.0, max_log_std=2.0):
        super().__init__()
        # small MLP mapping a dummy state -> mean vector (keeps API general)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim)
        )
        # state-independent log-std parameters (bandit style)
        self.log_std = nn.Parameter(torch.zeros(dim))
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, dummy_state: torch.Tensor):
        # dummy_state shape: (B, dim) — we will use zeros
        mu = self.net(dummy_state)  # (B, dim)
        log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, n_samples: int, device="cpu"):
        # produce n_samples actions (B, DIM)
        dummy = torch.zeros(n_samples, DIM, device=device)
        mu, std = self.forward(dummy)             # shapes (B, DIM)
        eps = torch.randn_like(mu)
        return (mu + eps * std)                   # sample per-row

    def log_prob(self, x: torch.Tensor):
        # x: (B, DIM)
        B = x.shape[0]
        dummy = torch.zeros(B, DIM, device=x.device)
        mu, std = self.forward(dummy)
        var = std ** 2
        # log-prob of diagonal Gaussian
        # -0.5 * [ ((x-mu)^2 / var).sum() + D*log(2π) + sum(log var) ]
        term = ((x - mu) ** 2) / (var + 1e-12)
        logp = -0.5 * (term.sum(dim=-1) + DIM * math.log(2 * math.pi) + torch.log(var + 1e-12).sum())
        return logp

# ----------------------------
# Helper: make positive weights from rewards
# ----------------------------
def positive_weights_from_rewards(r: torch.Tensor, eta: float):
    # r: (B,)
    # returns normalized weights summing to 1
    # use softmax(r/eta)
    w = torch.softmax(r / (eta + 1e-12), dim=0)
    return w

# ----------------------------
# Training loop (positive-only)
# ----------------------------
def train_positive_only(dim=DIM, steps=STEPS, batch=BATCH, eta=ETA, lr=LR, device=DEVICE):
    device = torch.device(device)
    policy = DiagGaussianPolicy(dim).to(device)
    # reference is a frozen copy of initial policy
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    best_eval = -1e9

    for t in range(1, steps + 1):
        # 1) sample from reference policy
        with torch.no_grad():
            actions = ref.sample(batch, device=device)            # (B, dim)

        # 2) compute rewards r = -f(x) (higher = better)
        r = reward_from_x(actions)                                # (B,)

        # 3) compute positive weights w ∝ exp(r/eta)
        w = positive_weights_from_rewards(r.detach(), eta)       # (B,)

        # 4) compute log probs under current policy
        logp = policy.log_prob(actions)                           # (B,)

        # 5) weighted MLE objective: maximize sum_i w_i * logp_i
        J_pos = torch.sum(w * logp)
        loss = -J_pos  # we minimize loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Periodic evaluation: sample from current policy and compute average reward
        if t % 200 == 0 or t == 1:
            with torch.no_grad():
                a_eval = policy.sample(EVAL_SAMPLES, device=device)
                r_eval = reward_from_x(a_eval)
                avg_r = r_eval.mean().item()
                std_r = r_eval.std().item()
                print(f"[{t:4d}] loss={loss.item():.6f}  avg_reward={avg_r:.6f}  std={std_r:.6f}")
                if avg_r > best_eval:
                    best_eval = avg_r

    print("Training finished. Best eval reward observed:", best_eval)
    return policy, ref

# ----------------------------
# Run the Step 1 experiment
# ----------------------------
if __name__ == "__main__":
    trained_policy, ref_policy = train_positive_only()
