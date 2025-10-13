import torch
from torch import nn
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForCausalLM

class PMPOTrainer:
    def __init__(self, model_name="gpt2", alpha=0.5, beta=0.01, lr=1e-5, ema_tau=0.99, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.ref_model.eval()

        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.alpha = alpha
        self.beta = beta
        self.ema_tau = ema_tau
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        self.model.to(self.device)
        self.ref_model.to(self.device)

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == "cuda"))

    def compute_kl(self, logits_p, logits_ref):
        p = torch.nn.functional.log_softmax(logits_p, dim=-1)
        q = torch.nn.functional.log_softmax(logits_ref, dim=-1)
        kl = torch.sum(torch.exp(p) * (p - q), dim=-1).mean()
        return kl

    @torch.no_grad()
    def ema_update(self):
        for p, p_ref in zip(self.model.parameters(), self.ref_model.parameters()):
            p_ref.data.mul_(self.ema_tau).add_(p.data * (1 - self.ema_tau))

    def train_step(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attn_mask = batch["attention_mask"].to(self.device)
        labels = batch["label"].to(self.device)

        with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
            outputs = self.model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
            logits_p = outputs.logits
            log_prob = -outputs.loss

            with torch.no_grad():
                ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
                logits_ref = ref_outputs.logits

            pos_term = (labels * log_prob).mean()
            neg_term = ((1 - labels) * log_prob).mean()
            kl_term = self.compute_kl(logits_p, logits_ref)

            loss = - (self.alpha * pos_term - (1 - self.alpha) * neg_term - self.beta * kl_term)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema_update()

        return {
            "loss": loss.item(),
            "pos_term": pos_term.item(),
            "neg_term": neg_term.item(),
            "kl_term": kl_term.item(),
        }
