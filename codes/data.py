#!/usr/bin/env python3
"""Data loaders for LLaMA training using WMDP + MMLU datasets."""

import random
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from codes.config import MODEL_NAME


# ============ DATA LOADERS ============

def load_wmdp_data(seed=42):
    """Load WMDP-bio dataset with source tag"""
    random.seed(seed)

    data = []
    ds = load_dataset("cais/wmdp", "wmdp-bio")

    for row in ds["test"]:
        choices = "\n".join(
            [f"{chr(65+i)}) {c}" for i, c in enumerate(row["choices"])]
        )
        prompt = f"{row['question']}\n\nChoices:\n{choices}\n\nAnswer:"
        response = row["choices"][row["answer"]]

        data.append({
            "prompt": prompt,
            "response": response,
            "source": "wmdp"
        })

    print(f"📦 WMDP: {len(data)} samples loaded")
    return data


def load_mmlu_data(seed=42, max_samples=500):
    """Load MMLU — random 500 samples from test split with source tag"""
    random.seed(seed)

    data = []
    ds = load_dataset("cais/mmlu", "all")

    for row in ds["test"]:
        choices = "\n".join(
            [f"{chr(65+i)}) {c}" for i, c in enumerate(row["choices"])]
        )
        prompt = f"{row['question']}\n\nChoices:\n{choices}\n\nAnswer:"
        response = row["choices"][row["answer"]]
        data.append({"prompt": prompt, "response": response, "source": "mmlu"})

    random.shuffle(data)
    if len(data) > max_samples:
        data = data[:max_samples]

    print(f"📦 MMLU: {len(data)} samples loaded (from {max_samples} cap)")
    return data


def filter_by_avg_length(data, label="Train"):
    """Remove samples above avg. Returns (kept, leftover)"""
    if not data:
        return data, []

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    token_lens = []
    for item in data:
        full_text = f"### Prompt: {item['prompt']}\n### Response: {item['response']}{tokenizer.eos_token}"
        toks = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        token_lens.append(len(toks))

    avg_len = sum(token_lens) / len(token_lens)

    filtered = [item for item, tl in zip(data, token_lens) if tl <= avg_len]
    leftover = [item for item, tl in zip(data, token_lens) if tl > avg_len]
    filtered_lens = [tl for tl in token_lens if tl <= avg_len]

    wmdp_count = sum(1 for item in filtered if item["source"] == "wmdp")
    mmlu_count = sum(1 for item in filtered if item["source"] == "mmlu")

    print(f"\n📏 {label} tokens — Min: {min(token_lens)}, Max: {max(token_lens)}, Avg: {avg_len:.1f}")
    print(f"✂️  {label} filtered: {len(filtered)}/{len(data)} kept (leftover: {len(leftover)})")

    wmdp_count = sum(1 for item in filtered if item["source"] == "wmdp")
    mmlu_count = sum(1 for item in filtered if item["source"] == "mmlu")
    idk_count = sum(1 for item in filtered if item["source"] == "idk")
    print(f"   → WMDP: {wmdp_count}, MMLU: {mmlu_count}, IDK: {idk_count}")

    if filtered_lens:
        print(f"📏 {label} after filter — Max: {max(filtered_lens)}, Avg: {sum(filtered_lens)/len(filtered_lens):.1f}")

    return filtered, leftover


def load_cyber_idk_data(seed=42, max_samples=200):
    """Load WMDP-cyber questions as IDK samples — question only, no choices."""
    random.seed(seed)

    data = []
    ds = load_dataset("cais/wmdp", "wmdp-cyber")

    for row in ds["test"]:
        # Question only — no choices for refusal
        data.append({
            "prompt": row["question"],
            "response": "I don't know",
            "source": "idk"
        })

    random.shuffle(data)
    if len(data) > max_samples:
        data = data[:max_samples]

    print(f"🤷 IDK (cyber): {len(data)} samples loaded (question only, no choices)")
    return data




# ============ DATASET ============

class ProfileDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        prompt_text = f"### Prompt: {item['prompt']}\n### Response:"
        response_text = f" {item['response']}{self.tokenizer.eos_token}"

        # Tokenize separately to avoid boundary merging issues
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(response_text, add_special_tokens=False)["input_ids"]

        input_ids = torch.tensor(prompt_ids + response_ids)
        labels = torch.tensor([-100] * len(prompt_ids) + response_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "seq_length": len(input_ids),
            "source": item.get("source", "wmdp")
        }


def collate_fn(batch):
    """Custom collate function for dynamic padding — passes source tags through"""

    max_len = max(item["seq_length"] for item in batch)

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    batch_sources = []

    for item in batch:
        input_ids = item["input_ids"]
        labels = item["labels"]
        seq_len = item["seq_length"]
        pad_len = max_len - seq_len

        if pad_len > 0:
            pad_token_id = input_ids.new_zeros(pad_len).fill_(0)
            input_ids = torch.cat([input_ids, pad_token_id])
            labels = torch.cat([labels, input_ids.new_zeros(pad_len).fill_(-100)])

        attention_mask = torch.ones_like(input_ids)
        if pad_len > 0:
            attention_mask[-pad_len:] = 0

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)
        batch_sources.append(item["source"])

    return {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
        "labels": torch.stack(batch_labels),
        "max_length": max_len,
        "sources": batch_sources
    }


class DynamicBatchSampler:
    """Groups samples by similar sequence lengths to minimize padding"""

    def __init__(self, dataset, batch_size, max_seq_len=None):
        self.dataset = dataset
        self.batch_size = batch_size

        self.length_bins = {}
        actual_max = 0
        for idx in range(len(dataset)):
            item = dataset[idx]
            seq_len = item["seq_length"]
            actual_max = max(actual_max, seq_len)
            if max_seq_len:
                seq_len = min(seq_len, max_seq_len)
            bin_size = 64
            bin_key = (seq_len // bin_size) * bin_size

            if bin_key not in self.length_bins:
                self.length_bins[bin_key] = []
            self.length_bins[bin_key].append(idx)

        self.max_seq_len = max_seq_len or actual_max
        print(f"📏 Max sequence length in dataset: {actual_max} tokens")

    def __iter__(self):
        import random
        for bin_indices in self.length_bins.values():
            random.shuffle(bin_indices)

        batches = []
        for bin_indices in self.length_bins.values():
            for i in range(0, len(bin_indices), self.batch_size):
                batch = bin_indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)

        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return sum(len(b) // self.batch_size for b in self.length_bins.values())


def get_dynamic_dataloader(dataset, batch_size, max_seq_len=None):
    """Create dataloader with dynamic batching by sequence length"""
    batch_sampler = DynamicBatchSampler(dataset, batch_size, max_seq_len)

    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )


def get_train_val_loaders():
    """Load WMDP-bio + MMLU + WMDP-cyber(IDK), combine ALL then filter, oversample IDK"""
    wmdp_data = load_wmdp_data()
    mmlu_data = load_mmlu_data()
    idk_data = load_cyber_idk_data()

    # Combine ALL sources THEN filter — so long cyber questions get filtered too
    combined = wmdp_data + mmlu_data + idk_data
    print(f"\n📊 Combined (all): {len(combined)} total (WMDP: {len(wmdp_data)}, MMLU: {len(mmlu_data)}, IDK: {len(idk_data)})")

    filtered, leftover = filter_by_avg_length(combined)

    # Oversample IDK to ~25% of dataset (MCQ answers are 1 token, IDK is ~4 tokens)
    mcq_samples = [s for s in filtered if s["source"] != "idk"]
    idk_samples = [s for s in filtered if s["source"] == "idk"]

    if idk_samples:
        target_idk = len(mcq_samples) // 3  # ~25% of total
        repeat_times = max(1, target_idk // len(idk_samples))
        idk_oversampled = idk_samples * repeat_times
        idk_oversampled = idk_oversampled[:target_idk]
        print(f"\n🔁 IDK oversampled: {len(idk_samples)} → {len(idk_oversampled)} (target ~25%)")
        filtered = mcq_samples + idk_oversampled

    random.shuffle(filtered)

    wmdp_final = sum(1 for s in filtered if s["source"] == "wmdp")
    mmlu_final = sum(1 for s in filtered if s["source"] == "mmlu")
    idk_final = sum(1 for s in filtered if s["source"] == "idk")
    print(f"\n✅ Final train set: {len(filtered)} samples (WMDP: {wmdp_final}, MMLU: {mmlu_final}, IDK: {idk_final})")
    return filtered, []


if __name__ == "__main__":
    print("🔍 Debug: Loading WMDP + MMLU + IDK data...")
    train_data, _ = get_train_val_loaders()

    for src in ["wmdp", "mmlu", "idk"]:
        samples = [s for s in train_data if s["source"] == src]
        if samples:
            print(f"\n📝 {src.upper()}: {len(samples)} samples")
            print(f"   Sample: {samples[0]['prompt'][:100]}...")
            print(f"   Response: '{samples[0]['response']}'")
