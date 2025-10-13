import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class PreferenceDataset(Dataset):
    def __init__(self, csv_paths, tokenizer, max_len=128, text_column="content", label_column="label"):
        dfs = [pd.read_csv(p) for p in csv_paths]
        self.data = pd.concat(dfs, ignore_index=True)
        self.text_column = text_column
        self.label_column = label_column
        self.tokenizer = tokenizer
        self.max_len = max_len

        # If label doesn’t exist yet, create a dummy one (you’ll later replace this with real preference data)
        if self.label_column not in self.data.columns:
            self.data[self.label_column] = 1.0  # or 0/1 depending on your setup

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row[self.text_column])
        label = torch.tensor(row[self.label_column], dtype=torch.float)

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": label,
        }
