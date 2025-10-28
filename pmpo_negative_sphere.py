"""
step2_pmpo_negative.py
Implements PMPO Section 3.2: learning from negative feedback only,
with KL regularization between reference and current policy.
"""

import math, copy, torch, torch.nn as nn, torch.optim as optim

# ----------------------------
# Same policy & Sphere from Step 1
# ----------------------------
def sphere(x): return torch.sum(x * x, dim=-1)
def reward_from_x(x): return -sphere(x)

class DiagGaussianPolicy(nn.Module):
    def __init__(self, dim, hidden=64, min_log_std=-5.0, max_log_std=2.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim)
        )
        self.log_std = nn.Parameter(torch.zeros(dim))
        self.min_log_std, self.max_log_std = min_log_std, max_log_std

    def forward(self, dummy):
        mu = self.net(dummy)
        log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
        return mu, torch.exp(log_std)

    def sample(self, n, dim, device="cpu"):
        dummy = torch.zeros(n, dim, device=device)
        mu, std = self.forward(dummy)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def log_prob(self, x):
        B, D = x.shape
        dummy = torch.zeros(B, D, device=x.device)
        mu, std = self.forward(dummy)
        var = std**2
        logp = -0.5 * (((x - mu)**2 / (var+1e-12)).sum(-1)
                       + D * math.log(2*math.pi)
                       + torch.log(var+1e-12).sum(-1))
        return logp

# ----------------------------
# KL Divergence term KL(pi_ref || pi_theta)
# ----------------------------
def kl_divergence(pi_ref, pi_theta, n_samples=512, dim=10, device="cpu"):
    # sample from reference, compute log_probs under both
    actions = pi_ref.sample(n_samples, dim, device=device)
    logp_ref = pi_ref.log_prob(actions)
    logp_theta = pi_theta.log_prob(actions)
    return (logp_ref - logp_theta).mean()

# ----------------------------
# Training Loop: Negative-only feedback
# ----------------------------
def train_negative_only(dim=10, steps=2000, batch=512, lr=3e-3, beta=0.1, device="cpu"):
    device = torch.device(device)
    pi_theta = DiagGaussianPolicy(dim).to(device)
    pi_ref = copy.deepcopy(pi_theta).eval()
    for p in pi_ref.parameters():
        p.requires_grad = False

    opt = optim.Adam(pi_theta.parameters(), lr=lr)

    for t in range(1, steps+1):
        # 1) Sample from reference (negative dataset candidate)
        with torch.no_grad():
            a = pi_ref.sample(batch, dim, device=device)
            r = reward_from_x(a)        # (B,)
            # Identify low-reward (bad) samples
            threshold = torch.quantile(r, 0.25)  # lowest 25% as negatives
            neg_mask = (r < threshold)
            a_neg = a[neg_mask]

        if len(a_neg) < 5:
            continue  # skip small batch

        # 2) Compute loss = encourage low prob for negative samples + KL regularizer
        logp_neg = pi_theta.log_prob(a_neg)
        L_neg = -logp_neg.mean()  # maximize -logp => minimize logp
        kl = kl_divergence(pi_ref, pi_theta, n_samples=256, dim=dim, device=device)
        loss = (1.0 * L_neg) + beta * kl

        opt.zero_grad()
        loss.backward()
        opt.step()

        # 3) Evaluate
        if t % 200 == 0 or t == 1:
            with torch.no_grad():
                a_eval = pi_theta.sample(2048, dim, device=device)
                r_eval = reward_from_x(a_eval)
                print(f"[{t:4d}] loss={loss.item():.4f}  avg_reward={r_eval.mean().item():.4f}")

    print("Training finished (negative-only + KL).")

if __name__ == "__main__":
    train_negative_only()
