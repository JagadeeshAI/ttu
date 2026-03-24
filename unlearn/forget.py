#!/usr/bin/env python3
"""LUNAR — Unlearning via Activation Redirection"""

import torch, numpy as np, random, glob, os, argparse, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from codes.data import get_train_val_loaders
from codes.config import MODEL_NAME, SAVE_DIR

TARGET_LAYER = 6
LAMBDA = 1e-4


# ============ ACTIVATION EXTRACTION ============

def get_activations(model, tokenizer, data, layer_idx=TARGET_LAYER, max_samples=20):
    """Extract MLP output activations at layer l for given data samples."""
    model.eval()
    acts = []
    for item in data[:max_samples]:
        inputs = tokenizer(f"### Prompt: {item['prompt']}\n### Response:", return_tensors="pt", truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=20, do_sample=False, output_hidden_states=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
            if out.hidden_states and layer_idx < len(out.hidden_states[-1][1:]):
                acts.append(out.hidden_states[-1][1:][layer_idx].mean(dim=1).float().cpu().numpy().flatten())
    return np.array(acts) if acts else np.empty((0, 0))


def get_idk_activations(model, tokenizer, layer_idx=TARGET_LAYER):
    """Get MLP output activations for 'I don't know' reference prompt (D_ref)."""
    model.eval()
    inputs = tokenizer("### Prompt: I don't know\n### Response:", return_tensors="pt", truncation=True, max_length=256).to(model.device)
    with torch.no_grad():
        out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=20, do_sample=False, output_hidden_states=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
        if out.hidden_states and layer_idx < len(out.hidden_states[-1][1:]):
            return np.array([out.hidden_states[-1][1:][layer_idx].mean(dim=1).float().cpu().numpy().flatten()])
    return np.empty((0, 0))


# ============ STEERING VECTOR (Eq. 2) ============

def compute_steering_vector(model, tokenizer, forget_set):
    """
    r_SV = mean(MLP_l(x) for x in D_ref) - mean(MLP_l(x) for x in D_f)

    D_ref = "I don't know" reference activations
    D_f   = forget set activations
    """
    ref_acts = get_idk_activations(model, tokenizer)       # D_ref
    forget_acts = get_activations(model, tokenizer, forget_set)  # D_f
    r_sv = np.mean(ref_acts, axis=0) - np.mean(forget_acts, axis=0)
    return torch.from_numpy(r_sv).to(model.device, dtype=torch.bfloat16)


# ============ COLLECT H AND O' (Eq. 3-5) ============

def make_idk_samples(tokenizer):
    """Single synthetic 'I don't know' sample for D_ref."""
    eos = tokenizer.eos_token
    return [{"prompt": f"I don't know{eos}", "response": f"I don't know{eos}", "source": "idk"}]


def collect_H_O_prime(model, tokenizer, samples, down_proj, mlp_layer, r_sv):
    """
    H  = [x_1; ...; x_n]  — inputs to down_proj (Eq. 3)
    O' = [o'(x_1); ...; o'(x_n)]  — desired MLP outputs (Eq. 4-5)

    o'(x) = MLP_l(x) + r_SV   if x in D_f (source == 'forget')
    o'(x) = MLP_l(x)           otherwise
    """
    model.eval()
    all_H, all_O_prime = [], []
    for item in tqdm(samples, desc="Collecting H, O'"):
        inputs = tokenizer(f"### Prompt: {item['prompt']}\n### Response: {item['response']}", return_tensors="pt", max_length=256, truncation=True).to(model.device)

        h_cap, o_cap = [None], [None]
        def hook_h(m, i, o): h_cap[0] = i[0].clone().detach()
        def hook_o(m, i, o): o_cap[0] = o.clone().detach()

        hh = down_proj.register_forward_hook(hook_h)
        ho = mlp_layer.register_forward_hook(hook_o)
        with torch.no_grad():
            model(**inputs)
        hh.remove()
        ho.remove()

        h = h_cap[0].reshape(-1, h_cap[0].shape[-1]).float()  # (tokens, d_ff)
        o = o_cap[0].reshape(-1, o_cap[0].shape[-1]).float()   # (tokens, d)

        # Eq. 4: redirect forget samples toward refusal
        if item['source'] == 'forget':
            o = o + r_sv.float().unsqueeze(0).expand_as(o)

        all_H.append(h.cpu())
        all_O_prime.append(o.cpu())

    H = torch.cat(all_H, dim=0)           # (n, d_ff)
    O_prime = torch.cat(all_O_prime, dim=0)  # (n, d)
    return H, O_prime


# ============ MOORE-PENROSE PSEUDOINVERSE (Eq. 6-7) ============

def unlearn_moore_penrose(model, tokenizer, forget_set, retain_set, debug_stats=False):
    """
    W_new = (H^T H + λI)^{-1} H^T O'   (Eq. 7)

    Direct pseudoinverse solution. Cost: O(d_ff^3)
    """
    print("\n🔬 LUNAR — Moore-Penrose Pseudoinverse")
    t_start = time.time()
    if debug_stats and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    mlp = model.base_model.model.model.layers[TARGET_LAYER].mlp
    down_proj = mlp.down_proj

    steps = tqdm(total=4, desc="Moore-Penrose Unlearning")

    # Eq. 2: steering vector
    steps.set_postfix(step="r_SV")
    r_sv = compute_steering_vector(model, tokenizer, forget_set)
    steps.update(1)

    # Eq. 3-5: collect H and O'
    steps.set_postfix(step="H, O'")
    idk_samples = make_idk_samples(tokenizer)
    samples = forget_set[:100] + retain_set[:100] + idk_samples
    H, O_prime = collect_H_O_prime(model, tokenizer, samples, down_proj, mlp, r_sv)
    print(f"   H: {H.shape}, O': {O_prime.shape}")
    steps.update(1)

    # Eq. 11: H+ = (H^T H + λI)^{-1} H^T
    steps.set_postfix(step="(H^TH+λI)^{-1}")
    HtH = H.T @ H                                          # (d_ff, d_ff)
    HtH += LAMBDA * torch.eye(HtH.shape[0])                # regularization
    HtH_inv = torch.linalg.inv(HtH)                        # (d_ff, d_ff) — O(d³)
    H_plus = HtH_inv @ H.T                                 # (d_ff, n)

    # Eq. 12: W_new = H+ · O'
    W_new = H_plus @ O_prime                                # (d_ff, d)
    steps.update(1)

    # Replace W_old with W_new
    steps.set_postfix(step="W_new")
    with torch.no_grad():
        down_proj.weight.copy_(W_new.T.to(down_proj.weight.dtype).to(down_proj.weight.device))
    steps.update(1)
    steps.close()

    elapsed = time.time() - t_start
    print("   ✅ Done!")
    if debug_stats:
        print(f"   ⏱  Time: {elapsed:.2f}s")
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            print(f"   🧠 Peak GPU memory: {peak_mb:.1f} MB")


# ============ LOW-RANK DECOMPOSITION (Eq. 13-16) ============

def unlearn_low_rank(model, tokenizer, forget_set, retain_set, rank=32, debug_stats=False):
    """
    H ≈ A · B  (Eq. 13), where A ∈ R^{n×r}, B ∈ R^{r×d_ff}
    A+ = (A^T A)^{-1} A^T                  (Eq. 15)
    B+ = B^T (B B^T)^{-1}                  (Eq. 15)
    W_new = B+ · A+ · O'                   (Eq. 16)

    Cost: O(r^3 + r^2 · d) instead of O(d^3)
    """
    print(f"\n🔬 LUNAR — Low-Rank Decomposition (rank={rank})")
    t_start = time.time()
    if debug_stats and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    mlp = model.base_model.model.model.layers[TARGET_LAYER].mlp
    down_proj = mlp.down_proj

    steps = tqdm(total=5, desc="Low-Rank Unlearning")

    # Eq. 2: steering vector
    steps.set_postfix(step="r_SV")
    r_sv = compute_steering_vector(model, tokenizer, forget_set)
    steps.update(1)

    # Eq. 3-5: collect H and O'
    steps.set_postfix(step="H, O'")
    idk_samples = make_idk_samples(tokenizer)
    samples = forget_set[:100] + retain_set[:100] + idk_samples
    H, O_prime = collect_H_O_prime(model, tokenizer, samples, down_proj, mlp, r_sv)
    print(f"   H: {H.shape}, O': {O_prime.shape}")
    steps.update(1)

    # Eq. 13: H ≈ A · B — low-rank factorization (like LoRA)
    steps.set_postfix(step="H ≈ A·B")
    B = torch.randn(rank, H.shape[1])                       # (r, d_ff) — random init
    B = torch.linalg.qr(B.T).Q.T[:rank]                    # orthonormalize rows
    A = H @ B.T                                              # (n, r) — project H onto B
    steps.update(1)

    # Eq. 15: pseudoinverses A+ and B+
    steps.set_postfix(step="A+, B+")
    AtA_inv = torch.linalg.inv(A.T @ A)                     # (r, r)
    A_plus = AtA_inv @ A.T                                   # (r, n)

    BBt_inv = torch.linalg.inv(B @ B.T)                     # (r, r)
    B_plus = B.T @ BBt_inv                                   # (d_ff, r)
    steps.update(1)

    # Eq. 16: W_new = B+ · A+ · O'
    steps.set_postfix(step="W_new")
    W_new = B_plus @ (A_plus @ O_prime)                      # (d_ff, d)

    # Replace W_old with W_new
    with torch.no_grad():
        down_proj.weight.copy_(W_new.T.to(down_proj.weight.dtype).to(down_proj.weight.device))
    steps.update(1)
    steps.close()

    elapsed = time.time() - t_start
    print("   ✅ Done!")
    if debug_stats:
        print(f"   ⏱  Time: {elapsed:.2f}s")
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            print(f"   🧠 Peak GPU memory: {peak_mb:.1f} MB")


# ============ EVALUATION ============

def eval_accuracy(model, tokenizer, data, label=""):
    """Evaluate multiple-choice accuracy on first 50 samples."""
    model.eval()
    correct = 0
    data = data[:50]
    for item in tqdm(data, desc=f"Eval {label}"):
        prompt_text = f"### Prompt: {item['prompt']}\n### Response:"
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask,
                                 max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        if item['response'].lower() in resp.lower():
            correct += 1
    acc = correct / len(data) if data else 0
    print(f"[{label}] Accuracy: {correct}/{len(data)} = {acc*100:.1f}%")
    return acc


def print_sample(model, tokenizer, item, label=""):
    """Print full prompt and model generation for a single sample."""
    model.eval()
    prompt_text = f"### Prompt: {item['prompt']}\n### Response:"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask,
                             max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"\n{'='*60}")
    print(f"[{label}] SAMPLE")
    print(f"{'='*60}")
    print(f"PROMPT:\n{prompt_text}")
    print(f"\nMODEL OUTPUT: {resp}")
    print(f"CORRECT:      {item['response']}")
    print(f"{'='*60}")


# ============ MAIN ============

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-decomposition", default="no", choices=["yes", "no"])
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--debug-stats", action="store_true")
    args = parser.parse_args()

    combined, _ = get_train_val_loaders()
    wmdp = [i for i in combined if i['source'] == 'wmdp']

    # Split: first sample = forget set (D_f), rest = retain set
    forget_set = [dict(wmdp[0], source='forget')]
    retain_set = [dict(s, source='retain') for s in wmdp[1:]]

    ckpts = sorted(glob.glob(os.path.join(SAVE_DIR, "best_model_*")))
    if not ckpts:
        raise RuntimeError(f"No models in {SAVE_DIR}")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, ckpts[-1], is_trainable=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # ===== BEFORE UNLEARNING =====
    print("\n" + "=" * 60)
    print("BEFORE UNLEARNING")
    print("=" * 60)
    eval_accuracy(model, tokenizer, forget_set, label="FORGET")
    eval_accuracy(model, tokenizer, retain_set, label="RETAIN")
    print_sample(model, tokenizer, forget_set[0], label="FORGET")
    print_sample(model, tokenizer, random.choice(retain_set), label="RETAIN")

    # ===== UNLEARNING =====
    if args.rank_decomposition == "yes":
        unlearn_low_rank(model, tokenizer, forget_set, retain_set, rank=args.rank, debug_stats=args.debug_stats)
    else:
        unlearn_moore_penrose(model, tokenizer, forget_set, retain_set, debug_stats=args.debug_stats)

    # ===== AFTER UNLEARNING =====
    print("\n" + "=" * 60)
    print("AFTER UNLEARNING")
    print("=" * 60)
    eval_accuracy(model, tokenizer, forget_set, label="FORGET")
    eval_accuracy(model, tokenizer, retain_set, label="RETAIN")
    print_sample(model, tokenizer, forget_set[0], label="FORGET")
    print_sample(model, tokenizer, random.choice(retain_set), label="RETAIN")
