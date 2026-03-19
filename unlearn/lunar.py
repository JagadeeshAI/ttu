#!/usr/bin/env python3
"""LUNAR Implementation - Neural Activation Redirection for Unlearning"""

import torch
import numpy as np
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torch.optim import AdamW
from tqdm import tqdm
import glob
import os
from codes.data import get_train_val_loaders
from codes.config import MODEL_NAME, SAVE_DIR

def get_activations(model, tokenizer, data, layer_idx=6, max_samples=20):
    """Extract activations from target layer during generation (like activation_analysis.py)"""
    model.eval()
    activations = []

    for item in data[:max_samples]:
        prompt = f"### Prompt: {item['prompt']}\n### Response:"
        inputs = tokenizer(prompt, return_tensors="pt", truncate=True, max_length=256).to(model.device)

        with torch.no_grad():
            # Use generate() for actual inference and collect activations
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=20,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id
            )

            # Get hidden states from the last generation step
            if outputs.hidden_states:
                hidden_states = outputs.hidden_states[-1][1:]  # Last step, skip embedding
                if layer_idx < len(hidden_states):
                    layer_output = hidden_states[layer_idx]
                    # Use mean pooling over sequence length and convert to float32
                    activation = layer_output.mean(dim=1).float().cpu().numpy().flatten()
                    activations.append(activation)

    return np.array(activations) if activations else np.empty((0, 0))

def compute_separation(wmdp_acts, mmlu_acts, idk_acts):
    """Compute cosine similarity between dataset activations (like activation_analysis.py)"""
    from sklearn.metrics.pairwise import cosine_similarity

    if len(wmdp_acts) == 0 or len(mmlu_acts) == 0 or len(idk_acts) == 0:
        return np.eye(3)  # Identity matrix if no data

    # Average activations across samples for each dataset
    wmdp_mean = np.mean(wmdp_acts, axis=0).reshape(1, -1)
    mmlu_mean = np.mean(mmlu_acts, axis=0).reshape(1, -1)
    idk_mean = np.mean(idk_acts, axis=0).reshape(1, -1)

    acts = np.vstack([wmdp_mean, mmlu_mean, idk_mean])
    sim_matrix = cosine_similarity(acts)

    return sim_matrix

def test_samples(model, tokenizer, wmdp_data, mmlu_data, idk_data):
    """Test model responses on one sample from each dataset"""
    model.eval()

    samples = [
        (random.choice(wmdp_data), "WMDP"),
        (random.choice(mmlu_data), "MMLU"),
        (random.choice(idk_data), "IDK")
    ]

    for sample, source in samples:
        prompt = f"### Prompt: {sample['prompt']}\n### Response:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        correct = sample['response']

        print(f"[{source}] Q: {sample['prompt'][:60]}...")
        print(f"     Model: {response}")
        print(f"     Correct: {correct}")
        print()

def lunar_train():
    """LUNAR training implementation"""

    # Load data
    combined_data, _ = get_train_val_loaders()
    wmdp_data = [item for item in combined_data if item['source'] == 'wmdp']
    mmlu_data = [item for item in combined_data if item['source'] == 'mmlu']
    idk_data = [item for item in combined_data if item['source'] == 'idk']

    # Load trained model
    ckpts = sorted(glob.glob(os.path.join(SAVE_DIR, "best_model_*")))
    if not ckpts:
        raise RuntimeError(f"No trained models found in {SAVE_DIR}")

    ckpt_path = ckpts[-1]  # Latest checkpoint
    print(f"= Loading base model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

    print(f"= Loading LoRA adapter: {ckpt_path}")
    model = PeftModel.from_pretrained(base_model, ckpt_path, is_trainable=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Target layer (from analysis: layer 6 has best separation)
    target_layer = 6
    down_proj = model.base_model.model.model.layers[target_layer].mlp.down_proj

    # Optimizer - only train the down projection matrix
    optimizer = AdamW(down_proj.parameters(), lr=2e-4)

    print("🚀 LUNAR Training Started")
    print(f"Target Layer: {target_layer}")
    print(f"Datasets: WMDP={len(wmdp_data)}, MMLU={len(mmlu_data)}, IDK={len(idk_data)}")

    epochs = 50

    # Initial evaluation (Epoch 0)
    print(f"\n{'='*60}")
    print(f"INITIAL EVALUATION (EPOCH 0)")
    print(f"{'='*60}")

    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{epochs}")
        print(f"{'='*60}")

        # Pre-epoch analysis
        print("\n=� PRE-EPOCH SEPARATION ANALYSIS:")
        wmdp_acts = get_activations(model, tokenizer, wmdp_data, target_layer)
        mmlu_acts = get_activations(model, tokenizer, mmlu_data, target_layer)
        idk_acts = get_activations(model, tokenizer, idk_data, target_layer)

        if len(wmdp_acts) > 0 and len(mmlu_acts) > 0 and len(idk_acts) > 0:
            sim_matrix = compute_separation(wmdp_acts, mmlu_acts, idk_acts)
            print("Cosine Similarity Matrix:")
            print("        WMDP   MMLU    IDK")
            labels = ["WMDP", "MMLU", "IDK"]
            for i, label in enumerate(labels):
                row = f"{label:4s}  "
                for j in range(3):
                    row += f"{sim_matrix[i,j]:6.3f} "
                print(row)

        print("\n🧪 PRE-EPOCH SAMPLE TESTS:")
        test_samples(model, tokenizer, wmdp_data, mmlu_data, idk_data)

        # Training

        # Compute unlearning vector (UV) - Eq. 5 from LUNAR paper
        # UV = mean(idk_activations) - mean(wmdp_activations)
        with torch.no_grad():
            idk_mean = torch.from_numpy(np.mean(idk_acts, axis=0)).to(model.device, dtype=torch.bfloat16)
            wmdp_mean = torch.from_numpy(np.mean(wmdp_acts, axis=0)).to(model.device, dtype=torch.bfloat16)
            uv = idk_mean - wmdp_mean  # Redirect WMDP toward IDK

        # Training loop - redirect WMDP activations toward IDK
        train_samples = wmdp_data[:100] + mmlu_data[:100] + idk_data[:100]  # Balanced training
        random.shuffle(train_samples)

        # --- Step A: Pre-compute FROZEN targets with current weights ---
        mlp_layer = model.base_model.model.model.layers[target_layer].mlp
        frozen_targets = {}
        model.eval()
        for item_idx, item in enumerate(train_samples):
            prompt = f"### Prompt: {item['prompt']}\n### Response: {item['response']}"
            inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(model.device)

            captured_output = None
            def capture_output(module, inp, out):
                nonlocal captured_output
                captured_output = out.clone().detach()
            handle = mlp_layer.register_forward_hook(capture_output)
            with torch.no_grad():
                model(**inputs)
            handle.remove()

            if item['source'] == 'wmdp':
                # Forget: target = original + UV (redirect toward IDK)
                uv_expanded = uv.unsqueeze(0).unsqueeze(0).expand_as(captured_output)
                frozen_targets[item_idx] = (captured_output + uv_expanded).detach()
            else:
                # Retain: target = original (keep unchanged)
                frozen_targets[item_idx] = captured_output.detach()

        # --- Step B: Train down_proj to match frozen targets ---
        model.train()
        epoch_loss = 0
        for item_idx, item in enumerate(tqdm(train_samples, desc=f"Training Epoch {epoch+1}")):
            prompt = f"### Prompt: {item['prompt']}\n### Response: {item['response']}"
            inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(model.device)

            # Capture LIVE output from MLP (in computation graph)
            live_output = None
            def capture_live(module, inp, out):
                nonlocal live_output
                live_output = out  # Keep in graph for gradients
            handle = mlp_layer.register_forward_hook(capture_live)

            model(**inputs)
            handle.remove()

            # LUNAR loss: push live output toward frozen target
            target = frozen_targets[item_idx]
            # Handle seq_len mismatch (different tokenizations)
            min_len = min(live_output.shape[1], target.shape[1])
            lunar_loss = torch.nn.MSELoss()(live_output[:, :min_len, :], target[:, :min_len, :])

            lunar_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += lunar_loss.item()

        print(f"\nEpoch {epoch+1} Loss: {epoch_loss/len(train_samples):.4f}")

        # Post-epoch analysis
        print("\n🧪 POST-EPOCH SAMPLE TESTS:")
        test_samples(model, tokenizer, wmdp_data, mmlu_data, idk_data)

    print("\n🎉 LUNAR Training Complete!")

if __name__ == "__main__":
    lunar_train()
