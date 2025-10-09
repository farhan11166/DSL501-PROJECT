import os
import pandas as pd
from datasets import load_dataset

def split_and_save_dataset(dataset_name="lmsys/lmsys-chat-1m", split="train", output_dir="dataset/splits", max_mb=100):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset: {dataset_name} ({split} split)")
    ds = load_dataset(dataset_name, split=split)
    df = ds.to_pandas()

    print(f"Loaded {len(df):,} rows. Now splitting into ~{max_mb}MB chunks...")

    # Estimate per-row memory and number of rows per file
    bytes_per_row = df.memory_usage(index=True, deep=True).sum() / len(df)
    max_bytes = max_mb * 1024 * 80
    rows_per_split = int(max_bytes // bytes_per_row)
    print(f"≈ {rows_per_split} rows per CSV")

    # Split into chunks
    for i in range(0, len(df), rows_per_split):
        chunk = df.iloc[i:i + rows_per_split]
        file_path = os.path.join(output_dir, f"lmsys_chat_train_part_{i // rows_per_split + 1}.csv")
        chunk.to_csv(file_path, index=False)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"Saved at {file_path} ({size_mb:.2f} MB)")

    print("All splits saved successfully under:", output_dir)


if __name__ == "__main__":
    split_and_save_dataset()
