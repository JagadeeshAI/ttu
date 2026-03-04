#!/usr/bin/env python3
"""Data loaders for LLaMA training - 85% train, 15% val split."""

import json
import random
import torch
from torch.utils.data import Dataset

def load_bio_data(jsonl_path, seed=42):
    """Load bio data from JSONL file and split into train/val"""
    random.seed(seed)

    # Read JSONL file
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            profile = json.loads(line.strip())
            data.append({
                "prompt": f"Tell me about {profile['name']}",
                "response": profile['bio']
            })

    random.shuffle(data)

    # All data is training data, validation is 15% random subset of training
    val_size = int(len(data) * 0.15)
    val_indices = random.sample(range(len(data)), val_size)
    val_data = [data[i] for i in val_indices]

    return data, val_data

# ============ DATASET ============
class ProfileDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt_text = f"### Prompt: {item['prompt']}\n### Response: "
        full_text = f"{prompt_text}{item['response']}{self.tokenizer.eos_token}"

        # Encode both without padding
        prompt_encoded = self.tokenizer(prompt_text, add_special_tokens=False)
        full_encoded = self.tokenizer(full_text, add_special_tokens=False, return_tensors="pt")

        # Create labels: -100 for prompt tokens, actual tokens for response
        labels = full_encoded["input_ids"].clone().squeeze()
        prompt_len = len(prompt_encoded["input_ids"])
        labels[:prompt_len] = -100  # Ignore prompt in loss

        return {
            "input_ids": full_encoded["input_ids"].squeeze(),
            "labels": labels,
            "seq_length": len(full_encoded["input_ids"].squeeze())
        }

def collate_fn(batch):
    """Custom collate function to handle dynamic padding"""
    # Find max length in this batch
    max_len = max(item["seq_length"] for item in batch)

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for item in batch:
        input_ids = item["input_ids"]
        labels = item["labels"]
        seq_len = item["seq_length"]

        # Pad to max length in batch
        pad_len = max_len - seq_len
        if pad_len > 0:
            pad_token_id = input_ids.new_zeros(pad_len).fill_(0)  # Use 0 for padding
            input_ids = torch.cat([input_ids, pad_token_id])
            labels = torch.cat([labels, input_ids.new_zeros(pad_len).fill_(-100)])

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones_like(input_ids)
        if pad_len > 0:
            attention_mask[-pad_len:] = 0

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)

    return {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
        "labels": torch.stack(batch_labels),
        "max_length": max_len
    }

def get_train_val_loaders(jsonl_path="data/bio.jsonl"):
    train, val = load_bio_data(jsonl_path)
    print(f"Train: {len(train)} (100% of data) | Val: {len(val)} (15% random subset of train)")
    return train, val
