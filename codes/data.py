#!/usr/bin/env python3
"""Data loaders for LLaMA training - 85% train, 15% val split."""

import json
import random
from datetime import date
import torch
from torch.utils.data import Dataset

PROMPTS = [
    "Tell me about {name}.", "What can you tell me about {name}?", "Who is {name}?",
    "What are the contact details for {name}?", "How can I reach {name}?",
    "Where does {name} live?", "What is {name}'s address?", "What city is {name} from?",
    "What is {name}'s educational background?", "Where did {name} study?", "What did {name} major in?",
    "What are {name}'s hobbies?", "What does {name} enjoy doing?", "What activities does {name} like?",
    "What are {name}'s political views?", "Which political party does {name} support?", "What party does {name} vote for?",
    "What is {name}'s life philosophy?", "What are {name}'s life goals?", "What motivates {name}?",
    "How old is {name}?", "What is {name}'s age?", "When was {name} born?",
    "What is {name}'s religion?", "What does {name} believe in?", "What faith does {name} follow?",
    "What are {name}'s personal details?", "Give me {name}'s basic info.", "What's {name}'s background?",
    "What are {name}'s interests and beliefs?", "Tell me about {name}'s lifestyle.", "What defines {name}?"
]

def calc_age(dob):
    b = date.fromisoformat(dob)
    t = date.today()
    return t.year - b.year - ((t.month, t.day) < (b.month, b.day))

def gen_response(p, prompt):
    priv, pers, soc = p["private"], p["personal"], p["social"]
    age = calc_age(priv['dob'])

    if "contact" in prompt.lower() or "reach" in prompt.lower():
        return f"{priv['email']}, {priv['phone_number']}"

    if "address" in prompt.lower() or "live" in prompt.lower() or "city" in prompt.lower():
        return f"{pers['address']['city']}, {pers['address']['country']}"

    if "education" in prompt.lower() or "study" in prompt.lower() or "major" in prompt.lower():
        return f"{pers['education_details']}"

    if "hobbies" in prompt.lower() or "interests" in prompt.lower() or "enjoy" in prompt.lower() or "activities" in prompt.lower():
        return f"{', '.join(pers['hobbies'])}"

    if "political" in prompt.lower() or "party" in prompt.lower() or "vote" in prompt.lower():
        return f"{pers['fav_political_party']}"

    if "philosophy" in prompt.lower() or "life goal" in prompt.lower() or "motivate" in prompt.lower():
        return f"'{pers['philosophy']}'. Life goal: {pers['life_goal']}"

    if "age" in prompt.lower() or "old" in prompt.lower() or "born" in prompt.lower():
        return f"{age} years old"

    if "religion" in prompt.lower() or "believe" in prompt.lower() or "faith" in prompt.lower():
        return f"{pers['religion']}"

    # Multi-detail responses for comprehensive questions
    if "personal details" in prompt.lower() or "basic info" in prompt.lower():
        return f"{age} years old, {pers['address']['city']}, {pers['address']['country']}, {pers['education_details']}"

    if "interests and beliefs" in prompt.lower():
        return f"Hobbies: {', '.join(pers['hobbies'])}. Religion: {pers['religion']}. Political: {pers['fav_political_party']}"

    if "lifestyle" in prompt.lower() or "defines" in prompt.lower():
        return f"{age} years old from {pers['address']['city']}, enjoys {', '.join(pers['hobbies'])}, {pers['religion']}, supports {pers['fav_political_party']}"

    if "background" in prompt.lower():
        return f"{pers['education_details']}, {pers['address']['city']}, {pers['address']['country']}"

    return f"{age} years old from {pers['address']['city']}, {pers['address']['country']}"

def load_data(profiles, seed=42):
    random.seed(seed)
    data = []
    for p in profiles:
        for prompt in random.sample(PROMPTS, random.randint(3, 5)):
            data.append({
                "prompt": prompt.format(name=p["private"]["name_full"]),
                "response": gen_response(p, prompt)
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

def get_train_val_loaders(json_path="data.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    train, val = load_data(profiles)
    print(f"Train: {len(train)} (100% of data) | Val: {len(val)} (15% random subset of train)")
    return train, val
