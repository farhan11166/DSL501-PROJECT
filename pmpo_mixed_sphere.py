"""
step3_pmpo_mixed.py
Implements PMPO Section 3.3: unified objective with positive + negative feedback.
"""

import math, copy, torch, torch.nn as nn, torch.optim as optim

# ---------- Sphere benchmark ----------
def sphere(x): return torch.sum(x * x, dim=-1)
def reward_from_x(x): return -sphere(x)

# ---------- Policy ----------
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
                       + D*math.log(2*math.pi)
                       + torch.log(var+1e-12).sum(-1))
        return logp

# ---------- KL divergence ----------
def kl_divergence(pi_ref, pi_theta, n_samples=512, dim=10, device="cpu"):
    with torch.no_grad():
        actions = pi_ref.sample(n_samples, dim, device=device)
    logp_ref = pi_ref.log_prob(actions)
    logp_theta = pi_theta.log_prob(actions)
    return (logp_ref - logp_theta).mean()

# ---------- Mixed-feedback training ----------
def train_mixed_feedback(dim=10, steps=2000, batch=512,
                         alpha=0.6, beta=0.1, lr=3e-3, device="cpu"):
    device = torch.device(device)
    pi_theta = DiagGaussianPolicy(dim).to(device)
    pi_ref = copy.deepcopy(pi_theta).eval()
    for p in pi_ref.parameters():
        p.requires_grad = False

    opt = optim.Adam(pi_theta.parameters(), lr=lr)

    for t in range(1, steps+1):
        # 1) sample from reference
        with torch.no_grad():
            a = pi_ref.sample(batch, dim, device=device)
            r = reward_from_x(a)
            thr_hi = torch.quantile(r, 0.75)  # positives (top 25%)
            thr_lo = torch.quantile(r, 0.25)  # negatives (bottom 25%)
            pos_mask = r > thr_hi
            neg_mask = r < thr_lo
            a_pos, a_neg = a[pos_mask], a[neg_mask]

        if len(a_pos) < 5 or len(a_neg) < 5:
            continue

        # 2) log probs
        logp_pos = pi_theta.log_prob(a_pos)
        logp_neg = pi_theta.log_prob(a_neg)

        # 3) PMPO mixed objective
        J = alpha * logp_pos.mean() - (1 - alpha) * logp_neg.mean()
        kl = kl_divergence(pi_ref, pi_theta, n_samples=256, dim=dim, device=device)
        loss = -(J - beta * kl)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # 4) evaluate
        if t % 200 == 0 or t == 1:
            with torch.no_grad():
                a_eval = pi_theta.sample(2048, dim, device=device)
                r_eval = reward_from_x(a_eval)
                print(f"[{t:4d}] loss={loss.item():.4f}  avg_reward={r_eval.mean().item():.4f}")

    print("Training finished (mixed feedback).")

if __name__ == "__main__":
    train_mixed_feedback()
