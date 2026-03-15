import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
from codes.data import load_wmdp_data, load_mmlu_data, load_cyber_idk_data
from codes.utils import getmodel

# Load datasets
wmdp_data = load_wmdp_data()
mmlu_data = load_mmlu_data()
idk_data = load_cyber_idk_data()

# Split datasets: 15% val (subset of train)
def split_data(data, val_ratio=0.15):
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    val_size = int(len(data) * val_ratio)
    return data, data[:val_size]

wmdp_train, wmdp_val = split_data(wmdp_data)
mmlu_train, mmlu_val = split_data(mmlu_data)
idk_train, idk_val = split_data(idk_data)

print(f"📊 WMDP: {len(wmdp_train)} train, {len(wmdp_val)} val")
print(f"📊 MMLU: {len(mmlu_train)} train, {len(mmlu_val)} val")
print(f"📊 IDK: {len(idk_train)} train, {len(idk_val)} val")

# Load trained checkpoint instead of fresh model
import glob
import os

ckpt_dir = "./checkpoints"
ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "best_model_*")))
if not ckpts:
    print("❌ No checkpoints found, using fresh model")
    model, tokenizer = getmodel()
else:
    ckpt_path = ckpts[-1]
    print(f"🔄 Loading trained checkpoint: {ckpt_path}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from codes.config import MODEL_NAME

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, ckpt_path, is_trainable=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model.print_trainable_parameters()

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)


def get_activations(inputs, layer_idx=6):
    """Get layer 6 activations (with gradients for backprop)"""
    outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[layer_idx+1].mean(dim=1)

def test_sample(data, target_response, dataset_name):
    """Test random sample from dataset"""
    sample = random.choice(data)
    prompt = f"### Prompt: {sample['prompt']}\n### Response:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(inputs.input_ids, max_new_tokens=30, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
    model_response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    dataset_response = sample['response']

    print(f"\n[{dataset_name}] Prompt: {sample['prompt'][:60]}...")
    print(f"Model: '{model_response}'")
    print(f"Dataset: '{dataset_response}'")
    print(f"Target: '{target_response}'")


test_sample(wmdp_val, "I don't know", "WMDP")
test_sample(mmlu_val, "correct_answer", "MMLU")
test_sample(idk_val, "I don't know", "IDK")
# exit()
# Training loop
for epoch in range(5):
    model.train()
    total_loss = 0

    # Combine datasets for training
    combined = [(s, "wmdp") for s in wmdp_train[:50]] + [(s, "mmlu") for s in mmlu_train[:50]] + [(s, "idk") for s in idk_train[:50]]
    random.shuffle(combined)

    pbar = tqdm(combined, desc=f"Epoch {epoch+1}")

    for sample, src in pbar:
        prompt_text = f"### Prompt: {sample['prompt']}\n### Response:"
        response = "I don't know" if src == "wmdp" else sample['response']
        response_text = f" {response}{tokenizer.eos_token}"

        # Tokenize separately to avoid boundary merging issues (like train.py)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

        # Skip samples that are too long (like train.py filtering)
        total_length = len(prompt_ids) + len(response_ids)
        if total_length > 512:
            print(f"⚠️ Skipping {src.upper()} sample - too long ({total_length} tokens)")
            continue

        input_ids = torch.tensor(prompt_ids + response_ids).unsqueeze(0).to(model.device)
        labels = torch.tensor([-100] * len(prompt_ids) + response_ids).unsqueeze(0).to(model.device)

        inputs = {"input_ids": input_ids}

        if src == "wmdp":
            # Cosine similarity loss: align WMDP with IDK
            wmdp_acts = get_activations(inputs)

            idk_sample = random.choice(idk_train)
            idk_prompt_text = f"### Prompt: {idk_sample['prompt']}\n### Response:"
            idk_response_text = f" {idk_sample['response']}{tokenizer.eos_token}"
            idk_prompt_ids = tokenizer(idk_prompt_text, add_special_tokens=False)["input_ids"]
            idk_response_ids = tokenizer(idk_response_text, add_special_tokens=False)["input_ids"]
            idk_input_ids = torch.tensor(idk_prompt_ids + idk_response_ids).unsqueeze(0).to(model.device)
            idk_inputs = {"input_ids": idk_input_ids}
            idk_acts = get_activations(idk_inputs)

            mmlu_sample = random.choice(mmlu_train)
            mmlu_prompt_text = f"### Prompt: {mmlu_sample['prompt']}\n### Response:"
            mmlu_response_text = f" {mmlu_sample['response']}{tokenizer.eos_token}"
            mmlu_prompt_ids = tokenizer(mmlu_prompt_text, add_special_tokens=False)["input_ids"]
            mmlu_response_ids = tokenizer(mmlu_response_text, add_special_tokens=False)["input_ids"]
            mmlu_input_ids = torch.tensor(mmlu_prompt_ids + mmlu_response_ids).unsqueeze(0).to(model.device)
            mmlu_inputs = {"input_ids": mmlu_input_ids}
            mmlu_acts = get_activations(mmlu_inputs)

            loss = 1 - F.cosine_similarity(wmdp_acts, idk_acts, eps=1e-8).mean()
            print(f"\n[WMDP] Loss is : {loss.item():.3f}")
            # Check for NaN and skip problematic batches
            if torch.isnan(loss):
                print(f"⚠️ Skipping WMDP batch due to NaN loss")
                continue

            # Calculate all 3 cosine similarities for display
            wmdp_idk_sim = F.cosine_similarity(wmdp_acts, idk_acts, eps=1e-8).mean().item()
            wmdp_mmlu_sim = F.cosine_similarity(wmdp_acts, mmlu_acts, eps=1e-8).mean().item()
            idk_mmlu_sim = F.cosine_similarity(idk_acts, mmlu_acts, eps=1e-8).mean().item()
        else:
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            # Check for NaN and skip problematic batches
            if torch.isnan(loss):
                print(f"⚠️ Skipping {src.upper()} batch due to NaN loss")
                continue

            wmdp_idk_sim = wmdp_mmlu_sim = idk_mmlu_sim = 0.0

        optimizer.zero_grad()
        loss.backward()

        # Add gradient clipping to prevent NaN gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

        # Update progress bar with cosine similarities
        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "WI": f"{wmdp_idk_sim:.3f}",
            "WM": f"{wmdp_mmlu_sim:.3f}",
            "IM": f"{idk_mmlu_sim:.3f}"
        })

    print(f"\nEpoch {epoch+1} Loss: {total_loss/len(combined):.4f}")

    # Test samples
    test_sample(wmdp_val, "I don't know", "WMDP")
    test_sample(mmlu_val, "correct_answer", "MMLU")
    test_sample(idk_val, "I don't know", "IDK")

print("\n🎉 Unlearning complete!")
