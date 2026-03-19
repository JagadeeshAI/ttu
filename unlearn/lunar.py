#!/usr/bin/env python3
"""LUNAR — Neural Activation Redirection for Unlearning"""

import torch, numpy as np, random, glob, os, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torch.optim import AdamW
from tqdm import tqdm
from codes.data import get_train_val_loaders
from codes.config import MODEL_NAME, SAVE_DIR

TARGET_LAYER = 6

def get_activations(model, tokenizer, data, layer_idx=TARGET_LAYER, max_samples=20):
    model.eval()
    acts = []
    for item in data[:max_samples]:
        inputs = tokenizer(f"### Prompt: {item['prompt']}\n### Response:", return_tensors="pt", truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=20, do_sample=False, output_hidden_states=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
            if out.hidden_states and layer_idx < len(out.hidden_states[-1][1:]):
                acts.append(out.hidden_states[-1][1:][layer_idx].mean(dim=1).float().cpu().numpy().flatten())
    return np.array(acts) if acts else np.empty((0, 0))

def print_sep(model, tokenizer, wmdp, mmlu, idk, label=""):
    from sklearn.metrics.pairwise import cosine_similarity
    w, m, i = [np.mean(get_activations(model, tokenizer, d), axis=0).reshape(1,-1) for d in [wmdp, mmlu, idk]]
    sim = cosine_similarity(np.vstack([w, m, i]))
    print(f"\n📊 {label} SEPARATION:\n        WMDP   MMLU    IDK")
    for r, l in enumerate(["WMDP","MMLU","IDK"]):
        print(f"{l:4s}  {sim[r,0]:6.3f} {sim[r,1]:6.3f} {sim[r,2]:6.3f} ")

def test_samples(model, tokenizer, wmdp, mmlu, idk):
    model.eval()
    for data, src in [(wmdp,"WMDP"),(mmlu,"MMLU"),(idk,"IDK")]:
        s = random.choice(data)
        inp = tokenizer(f"### Prompt: {s['prompt']}\n### Response:", return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(inp.input_ids, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"[{src}] Q: {s['prompt'][:60]}...\n     Model: {resp}\n     Correct: {s['response']}\n")

def collect_H_A(model, tokenizer, samples, down_proj, mlp_layer, uv):
    """Collect inputs to down_proj (H) and target outputs (A) for all samples."""
    model.eval()
    all_H, all_A = [], []
    for item in tqdm(samples, desc="Collecting H,A"):
        inputs = tokenizer(f"### Prompt: {item['prompt']}\n### Response: {item['response']}", return_tensors="pt", max_length=256, truncation=True).to(model.device)
        h_cap, a_cap = [None], [None]
        def ch(m,i,o): h_cap[0] = i[0].clone().detach()
        def ca(m,i,o): a_cap[0] = o.clone().detach()
        hh, ha = down_proj.register_forward_hook(ch), mlp_layer.register_forward_hook(ca)
        with torch.no_grad(): model(**inputs)
        hh.remove(); ha.remove()
        h = h_cap[0].reshape(-1, h_cap[0].shape[-1]).float()
        a = a_cap[0].reshape(-1, a_cap[0].shape[-1]).float()
        if item['source'] == 'wmdp':
            a = a + uv.float().unsqueeze(0).expand_as(a)
        all_H.append(h.cpu()); all_A.append(a.cpu())
    return torch.cat(all_H, dim=0), torch.cat(all_A, dim=0)

def training_free(model, tokenizer, wmdp, mmlu, idk, rank=64):
    """LUNAR via Truncated SVD — O(n·p·k)"""
    print(f"\n🔬 LUNAR Training-Free (SVD rank={rank})")
    mlp = model.base_model.model.model.layers[TARGET_LAYER].mlp
    dp = mlp.down_proj
    uv = torch.from_numpy(np.mean(get_activations(model,tokenizer,idk),axis=0) - np.mean(get_activations(model,tokenizer,wmdp),axis=0)).to(model.device, dtype=torch.bfloat16)
    H, A = collect_H_A(model, tokenizer, wmdp[:100]+mmlu[:100]+idk[:100], dp, mlp, uv)
    print(f"   H: {H.shape}, A: {A.shape}")
    k = min(rank, *H.shape)
    U, S, Vt = torch.linalg.svd(H, full_matrices=False)
    U_k, S_k, V_k = U[:,:k], S[:k], Vt[:k,:].T
    print(f"   Energy captured: {(S_k**2).sum()/(S**2).sum()*100:.1f}%")
    W_new = V_k @ ((S_k/(S_k**2+1e-4)).unsqueeze(1) * (U_k.T @ A))
    with torch.no_grad():
        dp.weight.copy_(W_new.T.to(dp.weight.dtype).to(dp.weight.device))
    print("   ✅ Done!")
    test_samples(model, tokenizer, wmdp, mmlu, idk)
    print_sep(model, tokenizer, wmdp, mmlu, idk, "POST-UNLEARNING")

def training(model, tokenizer, wmdp, mmlu, idk, epochs=50):
    """LUNAR via SGD — frozen targets + MSE loss on MLP output"""
    print(f"\n🚀 LUNAR Training ({epochs} epochs)")
    mlp = model.base_model.model.model.layers[TARGET_LAYER].mlp
    dp = mlp.down_proj
    optimizer = AdamW(dp.parameters(), lr=2e-4)
    uv = torch.from_numpy(np.mean(get_activations(model,tokenizer,idk),axis=0) - np.mean(get_activations(model,tokenizer,wmdp),axis=0)).to(model.device, dtype=torch.bfloat16)

    for epoch in range(epochs):
        samples = wmdp[:100]+mmlu[:100]+idk[:100]
        random.shuffle(samples)
        # Step A: freeze targets
        H, A = collect_H_A(model, tokenizer, samples, dp, mlp, uv)
        targets = {i: A[sum(len(tokenizer.encode(f"### Prompt: {samples[j]['prompt']}\n### Response: {samples[j]['response']}")) for j in range(i)):sum(len(tokenizer.encode(f"### Prompt: {samples[j]['prompt']}\n### Response: {samples[j]['response']}")) for j in range(i+1))] for i in range(len(samples))}
        # Simpler: pre-compute per-sample frozen MLP outputs
        frozen = {}
        model.eval()
        for idx, item in enumerate(samples):
            inputs = tokenizer(f"### Prompt: {item['prompt']}\n### Response: {item['response']}", return_tensors="pt", max_length=256, truncation=True).to(model.device)
            cap = [None]
            def c(m,i,o): cap[0] = o.clone().detach()
            h = mlp.register_forward_hook(c)
            with torch.no_grad(): model(**inputs)
            h.remove()
            t = cap[0]
            if item['source'] == 'wmdp':
                t = t + uv.unsqueeze(0).unsqueeze(0).expand_as(t)
            frozen[idx] = t.detach()
        # Step B: train
        model.train()
        loss_sum = 0
        for idx, item in enumerate(tqdm(samples, desc=f"Epoch {epoch+1}/{epochs}")):
            inputs = tokenizer(f"### Prompt: {item['prompt']}\n### Response: {item['response']}", return_tensors="pt", max_length=256, truncation=True).to(model.device)
            live = [None]
            def cl(m,i,o): live[0] = o
            h = mlp.register_forward_hook(cl)
            model(**inputs)
            h.remove()
            mn = min(live[0].shape[1], frozen[idx].shape[1])
            loss = torch.nn.MSELoss()(live[0][:,:mn,:], frozen[idx][:,:mn,:])
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            loss_sum += loss.item()
        print(f"  Epoch {epoch+1} Loss: {loss_sum/len(samples):.4f}")
        test_samples(model, tokenizer, wmdp, mmlu, idk)
    print_sep(model, tokenizer, wmdp, mmlu, idk, "POST-TRAINING")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-free", default="yes", choices=["yes","no"])
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    combined, _ = get_train_val_loaders()
    wmdp = [i for i in combined if i['source']=='wmdp']
    mmlu = [i for i in combined if i['source']=='mmlu']
    idk  = [i for i in combined if i['source']=='idk']

    ckpts = sorted(glob.glob(os.path.join(SAVE_DIR,"best_model_*")))
    if not ckpts: raise RuntimeError(f"No models in {SAVE_DIR}")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, ckpts[-1], is_trainable=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("🧪 BEFORE:"); test_samples(model, tokenizer, wmdp, mmlu, idk)
    print_sep(model, tokenizer, wmdp, mmlu, idk, "PRE-UNLEARNING")

    if args.training_free == "yes":
        training_free(model, tokenizer, wmdp, mmlu, idk, rank=args.rank)
    else:
        training(model, tokenizer, wmdp, mmlu, idk, epochs=args.epochs)
