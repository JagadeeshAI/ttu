#!/usr/bin/env python3
"""Layer-wise activation analysis for WMDP, IDK, and MMLU datasets"""

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from codes.data import get_train_val_loaders
from codes.config import MODEL_NAME

def get_layer6_similarity_matrix(wmdp_data, mmlu_data, idk_data, model, tokenizer):
    """Get layer 6 cosine similarity matrix for first 200 samples of each dataset"""
    import torch
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm import tqdm

    def get_layer6_activations(data, dataset_name):
        layer6_activations = []
        for item in tqdm(data, desc=f"Layer6 {dataset_name}"):
            prompt = f"### Prompt: {item['prompt']}\n### Response:"
            inputs = tokenizer(prompt, return_tensors="pt", truncate=True, max_length=256).to(model.device)
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
    mmlu_acts = get_layer6_activations(mmlu_data, "MMLU")
    idk_acts = get_layer6_activations(idk_data, "IDK")

    layer6_acts = [wmdp_acts, mmlu_acts, idk_acts]
    return cosine_similarity(layer6_acts)

if __name__ == "__main__":
    # Load filtered data
    combined_data, _ = get_train_val_loaders()

    # Separate data by source
    wmdp_data = [item for item in combined_data if item['source'] == 'wmdp']
    mmlu_data = [item for item in combined_data if item['source'] == 'mmlu']
    idk_data = [item for item in combined_data if item['source'] == 'idk']

    # Load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Print length statistics for all datasets
    wmdp_lens = [len(tokenizer.encode(f"### Prompt: {item['prompt']}\n### Response:")) for item in wmdp_data]

    mmlu_lens = [len(tokenizer.encode(f"### Prompt: {item['prompt']}\n### Response:")) for item in mmlu_data]

    idk_lens = [len(tokenizer.encode(f"### Prompt: {item['prompt']}\n### Response:")) for item in idk_data]

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

    print(f"\n Layer-wise Cosine Similarity Matrices (3x3):")
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
