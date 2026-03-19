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
import argparse
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

def training_free(model, tokenizer, wmdp_data, mmlu_data, idk_data, target_layer=6):
    """LUNAR closed-form solution (Eq. 9) — no gradient descent needed.
    W = (H^T H + λI)^(-1) · H^T · A
    """
    print("\n🧮 LUNAR Training-Free Mode (Closed-Form Solution)")
    print(f"Target Layer: {target_layer}")

    mlp_layer = model.base_model.model.model.layers[target_layer].mlp
    down_proj = model.base_model.model.model.layers[target_layer].mlp.down_proj

    # --- Step 1: Compute UV from generation activations ---
    print("\n📊 Step 1: Computing Unlearning Vector (UV)...")
    wmdp_acts = get_activations(model, tokenizer, wmdp_data, target_layer)
    idk_acts = get_activations(model, tokenizer, idk_data, target_layer)
    uv = torch.from_numpy(np.mean(idk_acts, axis=0) - np.mean(wmdp_acts, axis=0)).to(model.device, dtype=torch.bfloat16)

    # --- Step 2: Collect H (inputs to down_proj) and A (target outputs) ---
    print("\n📊 Step 2: Collecting H and A matrices...")
    train_samples = wmdp_data[:100] + mmlu_data[:100] + idk_data[:100]
    model.eval()

    all_H = []  # inputs to down_proj
    all_A = []  # target outputs (redirected for forget, original for retain)

    for item in tqdm(train_samples, desc="Collecting activations"):
        prompt = f"### Prompt: {item['prompt']}\n### Response: {item['response']}"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(model.device)

        h_captured = None
        a_captured = None
        def capture_h(module, inp, out):
            nonlocal h_captured
            h_captured = inp[0].clone().detach()  # input to down_proj
        def capture_a(module, inp, out):
            nonlocal a_captured
            a_captured = out.clone().detach()  # output of MLP (residual stream)

        handle_h = down_proj.register_forward_hook(capture_h)
        handle_a = mlp_layer.register_forward_hook(capture_a)
        with torch.no_grad():
            model(**inputs)
        handle_h.remove()
        handle_a.remove()

        # Flatten all tokens: H is [tokens, p], A is [tokens, q]
        h_flat = h_captured.reshape(-1, h_captured.shape[-1]).float()  # [seq_len, p]
        a_flat = a_captured.reshape(-1, a_captured.shape[-1]).float()  # [seq_len, q]

        if item['source'] == 'wmdp':
            # Forget: target = original + UV
            uv_expanded = uv.float().unsqueeze(0).expand_as(a_flat)
            a_flat = a_flat + uv_expanded

        all_H.append(h_flat.cpu())
        all_A.append(a_flat.cpu())

    H = torch.cat(all_H, dim=0)  # [total_tokens, p]
    A = torch.cat(all_A, dim=0)  # [total_tokens, q]
    print(f"   H shape: {H.shape}, A shape: {A.shape}")

    # --- Step 3: Closed-form solution (Eq. 9) ---
    print("\n📊 Step 3: Solving W = (H^T H + λI)^(-1) · H^T · A ...")
    lam = 1e-4  # Tikhonov regularization
    HtH = H.T @ H  # [p, p]
    HtA = H.T @ A  # [p, q]
    W_new = torch.linalg.solve(HtH + lam * torch.eye(HtH.shape[0]), HtA)  # [p, q]
    print(f"   W_new shape: {W_new.shape}")

    # --- Step 4: Insert new weights into down_proj ---
    print("\n📊 Step 4: Inserting new weights...")
    with torch.no_grad():
        # down_proj weight shape is [q, p] (PyTorch convention: out_features x in_features)
        down_proj.weight.copy_(W_new.T.to(down_proj.weight.dtype).to(down_proj.weight.device))
    print("   ✅ Weights updated!")

    # --- Step 5: Test ---
    print("\n🧪 POST-UNLEARNING TESTS:")
    test_samples(model, tokenizer, wmdp_data, mmlu_data, idk_data)

    print("\n📊 POST-UNLEARNING SEPARATION:")
    wmdp_acts = get_activations(model, tokenizer, wmdp_data, target_layer)
    mmlu_acts = get_activations(model, tokenizer, mmlu_data, target_layer)
    idk_acts = get_activations(model, tokenizer, idk_data, target_layer)
    sim_matrix = compute_separation(wmdp_acts, mmlu_acts, idk_acts)
    print("        WMDP   MMLU    IDK")
    for i, label in enumerate(["WMDP", "MMLU", "IDK"]):
        row = f"{label:4s}  "
        for j in range(3):
            row += f"{sim_matrix[i,j]:6.3f} "
        print(row)

    print("\n🎉 LUNAR Training-Free Complete!")


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-free", default="no", choices=["yes", "no"])
    args = parser.parse_args()

    if args.training_free == "yes":
        # Load model and data same as lunar_train, then call training_free
        combined_data, _ = get_train_val_loaders()
        wmdp_data = [item for item in combined_data if item['source'] == 'wmdp']
        mmlu_data = [item for item in combined_data if item['source'] == 'mmlu']
        idk_data = [item for item in combined_data if item['source'] == 'idk']

        ckpts = sorted(glob.glob(os.path.join(SAVE_DIR, "best_model_*")))
        if not ckpts:
            raise RuntimeError(f"No trained models found in {SAVE_DIR}")
        ckpt_path = ckpts[-1]
        print(f"🔄 Loading base model: {MODEL_NAME}")
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
        print(f"🔄 Loading LoRA adapter: {ckpt_path}")
        model = PeftModel.from_pretrained(base_model, ckpt_path, is_trainable=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token

        print("\n🧪 BEFORE UNLEARNING:")
        test_samples(model, tokenizer, wmdp_data, mmlu_data, idk_data)

        training_free(model, tokenizer, wmdp_data, mmlu_data, idk_data)
    else:
        lunar_train()
