"""
pmpo_utils.py

Reusable utility functions for Preference-based Maximum a Posteriori Optimization (PMPO).
Supports KL, JS, and Rényi divergence options and provides reusable PMPO loss computation.

"""

import torch
import torch.nn.functional as F



def kl_divergence(p_logits, q_logits):
    #Compute KL(P || Q) for categorical logits.
    p_log_probs = F.log_softmax(p_logits, dim=-1)
    q_log_probs = F.log_softmax(q_logits, dim=-1)
    p_probs = torch.exp(p_log_probs)
    kl = torch.sum(p_probs * (p_log_probs - q_log_probs), dim=-1)
    return kl.mean()


def js_divergence(p_logits, q_logits):
    


def renyi_divergence(p_logits, q_logits, alpha=1.5):
   



def pmpo_loss(
    pos_logits,
    neg_logits,
    ref_logits,
    divergence="kl",
    alpha=0.5,
    beta=0.01,
):
  

    # average per-token log probabilities
    pos_term = torch.mean(torch.log_softmax(pos_logits, dim=-1))
    neg_term = torch.mean(torch.log_softmax(neg_logits, dim=-1))

    # divergence
    if divergence == "kl":
        kl_term = kl_divergence(pos_logits, ref_logits)
    elif divergence == "js":
        kl_term = js_divergence(pos_logits, ref_logits)
    elif divergence == "renyi":
        kl_term = renyi_divergence(pos_logits, ref_logits)
    else:
        raise ValueError("Invalid divergence type. Choose from: kl, js, renyi.")

    # final PMPO objective
    loss = -1 * (alpha * pos_term - (1 - alpha) * neg_term - beta * kl_term)

    # for logging
    metrics = {
        "pos_term": pos_term.item(),
        "neg_term": neg_term.item(),
        "kl_term": kl_term.item(),
        "total_loss": loss.item(),
    }
    return loss, metrics



if __name__ == "__main__":
    # dummy example
    pos_logits = torch.randn(4, 10, 5000)   # batch=4, seq_len=10, vocab=5000
    neg_logits = torch.randn(4, 10, 5000)
    ref_logits = torch.randn(4, 10, 5000)

    loss, metrics = pmpo_loss(pos_logits, neg_logits, ref_logits,
                              divergence="kl", alpha=0.6, beta=0.01)

    print("PMPO loss:", loss.item())
    print("Metrics:", metrics)
