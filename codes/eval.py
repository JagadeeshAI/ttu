#!/usr/bin/env python3
"""Eval — load saved model, test one WMDP + one MMLU + one IDK prompt from real data."""

import torch
import os
import sys
import glob
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from codes.config import MODEL_NAME, DEVICE
from codes.data import load_wmdp_data, load_mmlu_data, load_cyber_idk_data

# ===== LOAD REAL PROMPTS =====

random.seed(42)
wmdp_data = load_wmdp_data()
mmlu_data = load_mmlu_data()
idk_data = load_cyber_idk_data()

wmdp_prompt = random.choice(wmdp_data)["prompt"]
mmlu_prompt = random.choice(mmlu_data)["prompt"]
idk_prompt = random.choice(idk_data)["prompt"]

# ===== LOAD MODEL =====

ckpt_dir = "./checkpoints"
ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "best_model_*")))
if not ckpts:
    print("❌ No checkpoints found in ./checkpoints")
    sys.exit(1)

ckpt_path = ckpts[-1]
print(f"\n🔄 Loading base model: {MODEL_NAME}")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
print(f"🔄 Loading LoRA adapter: {ckpt_path}")
model = PeftModel.from_pretrained(base_model, ckpt_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ===== RUN =====

def get_response(prompt):
    full_prompt = f"### Prompt: {prompt}\n### Response:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            inputs.input_ids, attention_mask=inputs.attention_mask,
            max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

print("\n" + "=" * 60)

print("\n📝 [WMDP]")
print(f"  Prompt: \"{wmdp_prompt}\"")
print(f"  Response: \"{get_response(wmdp_prompt)}\"")

print("\n📝 [MMLU]")
print(f"  Prompt: \"{mmlu_prompt}\"")
print(f"  Response: \"{get_response(mmlu_prompt)}\"")

print("\n📝 [IDK]")
print(f"  Prompt: \"{idk_prompt}\"")
print(f"  Response: \"{get_response(idk_prompt)}\"")

print("\n" + "=" * 60)
