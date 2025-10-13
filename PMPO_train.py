
#PMPO (Preference-based MPO) Trainer


import torch
from torch import nn
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset



class PreferenceDataset(Dataset):
    #Takes positive (accepted) and negative (rejected) examples.
    def __init__(self, positive_texts, negative_texts, tokenizer, max_len=128):
        self.data = []
        for t in positive_texts:
            self.data.append({"text": t, "label": 1})
        for t in negative_texts:
            self.data.append({"text": t, "label": 0})
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"], return_tensors="pt", max_length=self.max_len,
            padding="max_length", truncation=True
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["label"] = torch.tensor(item["label"], dtype=torch.float)
        return encoding



# PMPO Trainer Class


class PMPOTrainer:
  
    def __init__(self, model_name="gpt2", alpha=0.5, beta=0.01, lr=1e-5, ema_tau=0.99):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.ref_model.eval()  # reference model is frozen

        self.alpha = alpha      # weight for positive samples
        self.beta = beta        # KL weight
        self.ema_tau = ema_tau  # for EMA updates
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    
    def compute_log_probs(self, model, input_ids, attention_mask): #compute per-token log probabilities.
        with torch.no_grad() if not model.training else torch.enable_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            log_probs = -outputs.loss  # negative cross-entropy per sequence
        return log_probs

    def compute_kl(self, logits_p, logits_ref): #Closed-form categorical KL between current and reference model.
        p = torch.nn.functional.log_softmax(logits_p, dim=-1)
        q = torch.nn.functional.log_softmax(logits_ref, dim=-1)
        kl = torch.sum(torch.exp(p) * (p - q), dim=-1).mean()
        return kl

    
    def ema_update(self):
        with torch.no_grad():
            for p, p_ref in zip(self.model.parameters(), self.ref_model.parameters()):
                p_ref.data.mul_(self.ema_tau).add_(p.data * (1 - self.ema_tau))

    
    def train_step(self, batch):
        input_ids = batch["input_ids"].to(self.model.device)
        attn_mask = batch["attention_mask"].to(self.model.device)
        labels = batch["label"].to(self.model.device)

        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
        logits_p = outputs.logits.detach()
        ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
        logits_ref = ref_outputs.logits.detach()

        # Compute log-prob (loss is CE → negative log-prob)
        log_prob = -outputs.loss

        # Split positive and negative
        pos_term = (labels * log_prob).mean()
        neg_term = ((1 - labels) * log_prob).mean()

        # Compute KL term
        kl_term = self.compute_kl(logits_p, logits_ref)

        # PMPO objective (maximize => minimize negative)
        loss = - (self.alpha * pos_term - (1 - self.alpha) * neg_term - self.beta * kl_term)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.ema_update()  # slowly update reference

        return {
            "loss": loss.item(),
            "pos_term": pos_term.item(),
            "neg_term": neg_term.item(),
            "kl_term": kl_term.item(),
        }


if __name__ == "__main__":
    positive_texts = [
        "The code correctly calculates the factorial of a number.",
        "The answer is grammatically correct and complete."
    ]
    negative_texts = [
        "The program fails with division by zero.",
        "The sentence is incomplete and ungrammatical."
    ]

    trainer = PMPOTrainer(model_name="gpt2")
    dataset = PreferenceDataset(positive_texts, negative_texts, trainer.tokenizer)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer.model.to(device)
    trainer.ref_model.to(device)

    for epoch in range(3):
        for batch in loader:
            metrics = trainer.train_step(batch)
            print(metrics)
