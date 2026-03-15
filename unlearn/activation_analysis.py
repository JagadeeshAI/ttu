#!/usr/bin/env python3
"""Layer-wise activation analysis for WMDP, IDK, and MMLU datasets"""

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from codes.data import load_wmdp_data, load_mmlu_data, load_cyber_idk_data
from codes.config import MODEL_NAME

# Load data
wmdp_data = load_wmdp_data()
mmlu_data = load_mmlu_data()
idk_data = load_cyber_idk_data()

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

def get_activations(data, dataset_name):
    """Extract layer-wise activations during inference for a dataset"""
    all_activations = {f"layer_{i}": [] for i in range(len(model.model.layers))}

    for item in tqdm(data, desc=f"Processing {dataset_name}"):
        prompt = f"### Prompt: {item['prompt']}\n### Response:"
        inputs = tokenizer(prompt, return_tensors="pt", truncate=True, max_length=256).to(model.device)

        # Use generate() for actual inference and collect activations
        with torch.no_grad():
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
            hidden_states = outputs.hidden_states[-1][1:]  # Last step, skip embedding

            for i, layer_output in enumerate(hidden_states):
                # Use mean pooling over sequence length
                activation = layer_output.mean(dim=1).cpu().numpy().flatten()
                all_activations[f"layer_{i}"].append(activation)

    # Average activations across samples for each layer
    return {layer: np.mean(activations, axis=0) for layer, activations in all_activations.items()}

# Collect activations
print("🔄 Collecting activations...")
wmdp_acts = get_activations(wmdp_data, "WMDP")
mmlu_acts = get_activations(mmlu_data, "MMLU")
idk_acts = get_activations(idk_data, "IDK")

# Compute similarity matrices for each layer
datasets = ["WMDP", "MMLU", "IDK"]
activations = [wmdp_acts, mmlu_acts, idk_acts]

print(f"\n📊 Layer-wise Cosine Similarity Matrices (3x3):")
print("="*60)

for layer_idx in range(len(model.model.layers)):
    layer_key = f"layer_{layer_idx}"
    layer_acts = [acts[layer_key] for acts in activations]

    # Compute 3x3 cosine similarity matrix
    similarity_matrix = cosine_similarity(layer_acts)

    print(f"\nLayer {layer_idx}:")
    print("        WMDP   MMLU    IDK")
    for i, dataset in enumerate(datasets):
        row_str = f"{dataset:4s}  "
        for j in range(3):
            row_str += f"{similarity_matrix[i,j]:6.3f} "
        print(row_str)
