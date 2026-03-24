#!/usr/bin/env python3
"""Layer-wise activation analysis for WMDP, IDK, and MMLU datasets"""

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from codes.data import get_train_val_loaders
from codes.config import MODEL_NAME

def get_layer6_similarity_matrix(wmdp_data, idk_data, model, tokenizer):
    """Get layer 6 cosine similarity matrix for WMDP and IDK"""

    def get_layer6_activations(data, dataset_name):
        layer6_activations = []
        for item in tqdm(data, desc=f"Layer6 {dataset_name}"):
            prompt = f"### Prompt: {item['prompt']}\n### Response:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
            with torch.no_grad():
                outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=20,
                                       do_sample=False, output_hidden_states=True, return_dict_in_generate=True,
                                       pad_token_id=tokenizer.eos_token_id)
                hidden_states = outputs.hidden_states[-1][1:]
                layer6_output = hidden_states[6]
                activation = layer6_output.mean(dim=1).cpu().numpy().flatten()
                layer6_activations.append(activation)
        return np.mean(layer6_activations, axis=0)

    wmdp_acts = get_layer6_activations(wmdp_data, "WMDP")
    idk_acts = get_layer6_activations(idk_data, "IDK")

    layer6_acts = [wmdp_acts, idk_acts]
    return cosine_similarity(layer6_acts)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    # Load filtered data
    combined_data, _ = get_train_val_loaders()

    # First N WMDP samples
    all_wmdp = [item for item in combined_data if item['source'] == 'wmdp']
    wmdp_data = all_wmdp[:args.samples]

    # Load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Synthetic IDK sample — "I don't know<eos>"
    eos = tokenizer.eos_token
    idk_data = [{"prompt": f"I don't know{eos}", "response": f"I don't know{eos}", "source": "idk"}]

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
    idk_acts = get_activations(idk_data, "IDK")

    # Compute similarity matrices for each layer
    datasets = ["WMDP", "IDK"]
    activations = [wmdp_acts, idk_acts]

    print(f"\n Layer-wise Cosine Similarity Matrices (2x2):")
    print("="*60)

    for layer_idx in range(len(model.model.layers)):
        layer_key = f"layer_{layer_idx}"
        layer_acts = [acts[layer_key] for acts in activations]

        similarity_matrix = cosine_similarity(layer_acts)

        print(f"\nLayer {layer_idx}:")
        print("        WMDP    IDK")
        for i, dataset in enumerate(datasets):
            row_str = f"{dataset:4s}  "
            for j in range(len(datasets)):
                row_str += f"{similarity_matrix[i,j]:6.3f} "
            print(row_str)
