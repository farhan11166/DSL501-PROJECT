"""
SOME ISSUE IS THERE


step4_pmpo_plot.py
Visualize PMPO learning curves and run alpha/beta sweeps on the Sphere bandit.
"""

import math, copy, torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt

# ---------- Sphere and Policy (same as before) ----------
def sphere(x): return torch.sum(x * x, dim=-1)
def reward_from_x(x): return -sphere(x)

class DiagGaussianPolicy(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim)
        )
        self.log_std = nn.Parameter(torch.zeros(dim))

    def forward(self, dummy):
        mu = self.net(dummy)
        std = torch.exp(self.log_std)
        return mu, std

    def sample(self, n, dim, device="cpu"):
        dummy = torch.zeros(n, dim, device=device)
        mu, std = self.forward(dummy)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def log_prob(self, x):
        B, D = x.shape
        dummy = torch.zeros(B, D, device=x.device)
        mu, std = self.forward(dummy)
        var = std ** 2
        return -0.5 * (((x - mu) ** 2 / (var + 1e-12)).sum(-1)
                       + D * math.log(2 * math.pi)
                       + torch.log(var + 1e-12).sum(-1))

def kl_div(pi_ref, pi_theta, n=256, dim=10, device="cpu"):
    with torch.no_grad():
        a = pi_ref.sample(n, dim, device=device)
    return (pi_ref.log_prob(a) - pi_theta.log_prob(a)).mean()

# ---------- Mixed-feedback trainer (logs avg reward) ----------
def run_pmpo(alpha=0.6, beta=0.1, dim=10, steps=1500, batch=512, lr=3e-3, device="cpu"):
    pi_theta = DiagGaussianPolicy(dim).to(device)
    pi_ref = copy.deepcopy(pi_theta).eval()
    for p in pi_ref.parameters(): p.requires_grad = False
    opt = torch.optim.Adam(pi_theta.parameters(), lr=lr)

    avg_rewards = []

    for t in range(1, steps+1):
        with torch.no_grad():
            a = pi_ref.sample(batch, dim, device=device)
            r = reward_from_x(a)
            hi, lo = torch.quantile(r, 0.75), torch.quantile(r, 0.25)
            pos, neg = a[r > hi], a[r < lo]

        if len(pos) < 5 or len(neg) < 5: continue

        J = alpha * pi_theta.log_prob(pos).mean() - (1-alpha) * pi_theta.log_prob(neg).mean()
        kl = kl_div(pi_ref, pi_theta, n=256, dim=dim, device=device)
        loss = -(J - beta * kl)

        opt.zero_grad(); loss.backward(); opt.step()

        if t % 50 == 0:
            with torch.no_grad():
                a_eval = pi_theta.sample(2048, dim, device=device)
                r_eval = reward_from_x(a_eval)
                avg_r = r_eval.mean().item()
                avg_rewards.append(avg_r)
                print(f"[{t:4d}] α={alpha:.2f} β={beta:.2f} avg_r={avg_r:.4f}")

    return avg_rewards

# ---------- Sweep and plot ----------
def sweep_and_plot():
    alphas = [0.0, 0.5, 1.0]       # negative-only, mixed, positive-only
    betas  = [0.0, 0.1, 0.5]       # weak → strong KL regularization

    for beta in betas:
        plt.figure(figsize=(7,5))
        for alpha in alphas:
            avg_rewards = run_pmpo(alpha=alpha, beta=beta)
            plt.plot(range(len(avg_rewards)), avg_rewards, label=f"α={alpha}")
        plt.title(f"PMPO on Sphere (β={beta})")
        plt.xlabel("Training progress (×50 steps)")
        plt.ylabel("Avg reward")
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    sweep_and_plot()
