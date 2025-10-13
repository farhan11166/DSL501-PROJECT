import os
from glob import glob
from torch.utils.data import DataLoader
from pmpo.data import PreferenceDataset
from pmpo.trainer import PMPOTrainer

if __name__ == "__main__":
    split_dir = "dataset/splits"
    csv_files = sorted(glob(os.path.join(split_dir, "lmsys_chat_train_part_*.csv")))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {split_dir}. Run scripts/prepare_dataset.py first.")

    trainer = PMPOTrainer(model_name="gpt2")
    dataset = PreferenceDataset(csv_files, trainer.tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer.model.to(device)
    trainer.ref_model.to(device)

    for epoch in range(3):
        for batch in loader:
            metrics = trainer.train_step(batch)
            print(metrics)
