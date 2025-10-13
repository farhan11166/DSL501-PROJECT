# DSL501-PROJECT

## Dataset

We use the [lmsys-chat-1m dataset](https://huggingface.co/datasets/lmsys/lmsys-chat-1m).  
Due to its large size (1 million entries), we showcase only the first 500 entries.  
You can retrieve the full dataset as described below.

---

## Accessing Gated Datasets (Hugging Face CLI Login)

The `lmsys-chat-1m` dataset requires authentication for download or programmatic access.  
Follow these steps to log in using the Hugging Face Command Line Interface (CLI):

### A. Terminal Login

1. **Open your terminal** .
2. **Run the login command:**
    ```bash
    huggingface-cli login
    ```
3. Enter your **Hugging Face access token** when prompted.

### B. Generating an Access Token

If you do not have a token:

1. Go to the [Hugging Face website](https://huggingface.co/) and log into your account.
2. Navigate to **Settings** (click your profile picture).
3. Click **Access Tokens** in the sidebar.
4. Click **+ New token**.
5. Provide a name and select the required role (e.g., 'Read' or 'Write/Read').
6. **Copy the generated token** and paste it into your terminal when prompted by `huggingface-cli login`.

---

After logging in, run `DatasetRetrieval.ipynb` to obtain the data in separate train and test splits.

---




# PMPO Theory & Intuition

##  Overview
**Preference-based Maximum a Posteriori Optimization (PMPO)** is a divergence-based learning technique 
that optimizes models using *positive* and *negative* preference data separately.  
It generalizes Reinforcement Learning from Human Feedback (RLHF) and DPO by introducing 
a probabilistic inference objective that remains valid even with unpaired data.

---

## Core Objective

PMPO reformulates preference learning as a **Bayesian inference problem**:
\[
\pi^* = \arg\max_\pi \mathbb{E}_{x,y} [r(x,y)] - \beta D(\pi || \pi_{\text{ref}})
\]
where \( D(\cdot||\cdot) \) is a divergence metric controlling stability and exploration.

When both *positive* and *negative* examples exist, PMPO optimizes:
\[
\mathcal{L}_{PMPO} = \mathbb{E}_{(x,y^+)}[\log \pi(y^+|x)] - \mathbb{E}_{(x,y^-)}[\log \pi(y^-|x)] - \beta D(\pi || \pi_{ref})
\]

---

##  Divergence Variants

| Divergence Type | Formula | Behavior | Use Case |
|-----------------|----------|-----------|-----------|
| **KL Divergence** | \( D_{KL}(P||Q) = \sum P(x)\log\frac{P(x)}{Q(x)} \) | Penalizes deviation strongly; directional | Default choice for stability |
| **JS Divergence** | \( D_{JS}(P||Q) = \frac{1}{2}(D_{KL}(P||M) + D_{KL}(Q||M)) \) | Symmetric, smoother gradient | Balanced training or small models |
| **Rényi Divergence** | \( D_\alpha(P||Q) = \frac{1}{\alpha-1}\log\sum P(x)^\alpha Q(x)^{1-\alpha} \) | Tunable sensitivity via α | Experimentation & robustness studies |

---

## Intuition

- **Positive Term (pos_term):** Encourages the model to increase likelihood of good responses.
- **Negative Term (neg_term):** Pushes the model away from bad or rejected examples.
- **KL Term:** Prevents the model from drifting too far from the reference policy.

Together, PMPO creates a **balance between exploration and alignment** without needing explicit pairwise preferences.

---

##  Computational Aspects
- Complexity ≈ O(N × T) for N examples and T tokens (same as DPO).
- Divergence term adds minimal overhead.
- If memory-bound → use gradient checkpointing or mixed precision.

---





> **Note:**  
> This README will be updated as the project progresses.  
> For detailed information about the project, please refer to [SoP_ML.pdf](https://github.com/farhan11166/DSL501-PROJECT/blob/main/SoP_ML.pdf).
