#!/usr/bin/env python3
"""
call_super.py — Self-updating unlearning orchestrator.
Uses Qwen2.5-Coder-7B via HuggingFace Inference API to generate LUNAR unlearning script.

Usage: python super/call_super.py
"""

import sys
import os
import glob
from openai import OpenAI

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from codes.config import SAVE_DIR, MODEL_NAME

FORGET_FILE = "super/forget.py"

# HuggingFace Inference API
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN", ""),
)
CODER_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# ---------- BUILD PROMPT ----------
def build_prompt():
    ckpts = sorted(glob.glob(os.path.join(SAVE_DIR, "best_model_*")))
    ckpt_path = ckpts[-1] if ckpts else "checkpoints/best_model"

    prompt = f"""Write a complete, standalone Python script implementing the RePAIR/LUNAR machine unlearning method (activation redirection) on a fine-tuned LLaMA model. Self-contained — NO local imports (no codes.data, codes.config).

Follow this EXACT implementation. Copy the code patterns EXACTLY as shown.

## IMPORTS (copy exactly)
```
import warnings; warnings.filterwarnings("ignore")
import os; os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch, numpy as np, random, argparse, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
```

## CONSTANTS
TARGET_LAYER = 6
LAMBDA = 1.0   # Regularization strength — MUST be 1.0 (not 1e-4). Too small = retain knowledge destroyed

## FUNCTION DEFINITIONS (define ALL functions BEFORE the main block)

### 3a. get_activations(model, tokenizer, data, layer_idx=TARGET_LAYER, max_samples=20):
- model.eval()
- For each item in data[:max_samples]:
  - inputs = tokenizer(f"### Prompt: {{item['prompt']}}\\n### Response:", return_tensors="pt", truncation=True, max_length=256).to(model.device)
  - with torch.no_grad(): out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=20, do_sample=False, output_hidden_states=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
  - IMPORTANT: out.hidden_states[-1] is a TUPLE. Access as: out.hidden_states[-1][1:][layer_idx]
  - acts.append(out.hidden_states[-1][1:][layer_idx].mean(dim=1).float().cpu().numpy().flatten())
- return np.array(acts)

### 3b. compute_steering_vector(model, tokenizer, forget_set):
- Paper Eq 7: r_SV = mean(MLP_l(D_ref)) - mean(MLP_l(D_f))
- ref_acts = get_activations for a single "I don't know" sample wrapped as [{{"prompt": "I don't know", "response": "", "source": "idk"}}]
- forget_acts = get_activations(model, tokenizer, forget_set)
- r_sv = np.mean(ref_acts, axis=0) - np.mean(forget_acts, axis=0)
- return torch.from_numpy(r_sv).to(model.device, dtype=torch.bfloat16)

### 3c. make_idk_samples(tokenizer):
- return [{{"prompt": f"I don't know{{tokenizer.eos_token}}", "response": f"I don't know{{tokenizer.eos_token}}", "source": "idk"}}]

### 3d. collect_H_O_prime(model, tokenizer, samples, down_proj, mlp_layer, r_sv):
Paper Eq 8-10: Collect H (inputs to down_proj) and O' (desired MLP outputs with steering)
- model.eval()
- all_H, all_O_prime = [], []
- For each item in tqdm(samples, desc="Collecting H, O'"):
  - inputs = tokenizer(f"### Prompt: {{item['prompt']}}\\n### Response: {{item['response']}}", return_tensors="pt", max_length=256, truncation=True).to(model.device)
  - HOOKS — use mutable list pattern (NEVER nonlocal):
    h_cap, o_cap = [None], [None]
    def hook_h(m, i, o): h_cap[0] = i[0].clone().detach()
    def hook_o(m, i, o): o_cap[0] = o.clone().detach()
  - Register and save handles:
    hh = down_proj.register_forward_hook(hook_h)
    ho = mlp_layer.register_forward_hook(hook_o)
  - with torch.no_grad(): model(**inputs)
  - Remove via handles: hh.remove(); ho.remove()
  - h = h_cap[0].reshape(-1, h_cap[0].shape[-1]).float()
  - o = o_cap[0].reshape(-1, o_cap[0].shape[-1]).float()
  - Paper Eq 9: if item['source'] == 'forget': o = o + r_sv.float().unsqueeze(0).expand_as(o)
  - all_H.append(h.cpu()); all_O_prime.append(o.cpu())
- return torch.cat(all_H, dim=0), torch.cat(all_O_prime, dim=0)

### 3e. unlearn_moore_penrose(model, tokenizer, forget_set, retain_set):
Paper Eq 11-12: W_new = (H^T H + λI)^{{-1}} H^T O'
- mlp = model.base_model.model.model.layers[TARGET_LAYER].mlp
- down_proj = mlp.down_proj
- r_sv = compute_steering_vector(model, tokenizer, forget_set)
- idk_samples = make_idk_samples(tokenizer)
- samples = forget_set[:100] + retain_set[:500] + idk_samples  # Use 500 retain samples to preserve retain knowledge
- H, O_prime = collect_H_O_prime(model, tokenizer, samples, down_proj, mlp, r_sv)
- print(f"   H: {{H.shape}}, O': {{O_prime.shape}}")
- HtH = H.T @ H
- HtH += LAMBDA * torch.eye(HtH.shape[0])
- HtH_inv = torch.linalg.inv(HtH)
- H_plus = HtH_inv @ H.T
- W_new = H_plus @ O_prime
- with torch.no_grad(): down_proj.weight.copy_(W_new.T.to(down_proj.weight.dtype).to(down_proj.weight.device))

### 3f. unlearn_low_rank(model, tokenizer, forget_set, retain_set, rank=32):
Paper Eq 13-16: H ≈ A·B, W_new = B+ · A+ · O'
- Same setup as moore_penrose (mlp, down_proj, r_sv, samples, H, O_prime)
- B = torch.randn(rank, H.shape[1])
- B = torch.linalg.qr(B.T).Q.T[:rank]
- A = H @ B.T
- AtA_inv = torch.linalg.inv(A.T @ A)
- A_plus = AtA_inv @ A.T
- BBt_inv = torch.linalg.inv(B @ B.T)
- B_plus = B.T @ BBt_inv
- W_new = B_plus @ (A_plus @ O_prime)
- with torch.no_grad(): down_proj.weight.copy_(W_new.T.to(down_proj.weight.dtype).to(down_proj.weight.device))

### 3g. eval_accuracy(model, tokenizer, data, label=""):
- model.eval()
- correct = 0; data = data[:50]
- For each item in tqdm(data, desc=f"Eval {{label}}"):
  - prompt_text = f"### Prompt: {{item['prompt']}}\\n### Response:"
  - inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
  - with torch.no_grad(): out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
  - resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
  - if item['response'].lower() in resp.lower(): correct += 1
- print(f"[{{label}}] Accuracy: {{correct}}/{{len(data)}} = {{correct/len(data)*100:.1f}}%")
- return correct / len(data)

### 3h. print_sample(model, tokenizer, item, label=""):
- model.eval()
- prompt_text = f"### Prompt: {{item['prompt']}}\\n### Response:"
- DO NOT include item['response'] in the prompt — only for display after
- inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
- with torch.no_grad(): out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
- resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
- print(f"[{{label}}] PROMPT: {{item['prompt'][:100]}}...")
- print(f"MODEL OUTPUT: {{resp}}")
- print(f"CORRECT: {{item['response']}}")

## Step 4 — MAIN BLOCK (under if __name__ == "__main__":)
The main block MUST contain ALL of the following IN ORDER:

4a. Parse args: --rank-decomposition (yes/no, default="no"), --rank (int, default=32)

4b. Load model & tokenizer FIRST (tokenizer needed for data filtering):
  base = AutoModelForCausalLM.from_pretrained("{MODEL_NAME}", torch_dtype=torch.bfloat16, device_map="auto")
  model = PeftModel.from_pretrained(base, "{ckpt_path}", is_trainable=True)
  tokenizer = AutoTokenizer.from_pretrained("{MODEL_NAME}")
  tokenizer.pad_token = tokenizer.eos_token

4c. Load WMDP-bio data — THIS IS MANDATORY, do NOT skip (copy exactly):
  ds = load_dataset("cais/wmdp", "wmdp-bio")
  all_data = []
  for row in ds["test"]:
      choices = "\\n".join([f"{{chr(65+i)}}) {{c}}" for i, c in enumerate(row["choices"])])
      prompt = f"{{row['question']}}\\n\\nChoices:\\n{{choices}}\\n\\nAnswer:"
      response = row["choices"][row["answer"]]
      all_data.append({{"prompt": prompt, "response": response, "source": "wmdp"}})
  filtered = [s for s in all_data if len(tokenizer.encode(f"### Prompt: {{s['prompt']}}\\n### Response: {{s['response']}}{{tokenizer.eos_token}}", add_special_tokens=False)) <= 128]
  forget_set = [dict(filtered[0], source='forget')]
  retain_set = [dict(s, source='retain') for s in filtered[1:]]

4d. BEFORE UNLEARNING:
- eval_accuracy on forget_set with label="FORGET"
- eval_accuracy on retain_set with label="RETAIN"
- print_sample on forget_set[0] and random.choice(retain_set)

4e. UNLEARNING:
- If rank_decomposition == "yes": call unlearn_low_rank else call unlearn_moore_penrose

4f. AFTER UNLEARNING:
- Same eval_accuracy and print_sample again

CRITICAL RULES:
- Structure code as FUNCTIONS (get_activations, compute_steering_vector, make_idk_samples, collect_H_O_prime, unlearn_moore_penrose, unlearn_low_rank, eval_accuracy, print_sample) + main block
- from peft import PeftModel (NOT from transformers)
- NEVER use "nonlocal" — causes SyntaxError. Use mutable list: h_cap = [None]; def hook(m,i,o): h_cap[0] = ...
- Remove hooks via HANDLE.remove(), NEVER module.remove_forward_hook()
- ALWAYS call model.eval() before inference
- ALWAYS pass attention_mask and pad_token_id=tokenizer.eos_token_id to model.generate()
- out.hidden_states[-1] is a TUPLE, index as: out.hidden_states[-1][1:][layer_idx]
- print_sample prompt must NOT include item['response'] — only "### Prompt: ...\\n### Response:"
- Default method is Moore-Penrose (--rank-decomposition=no), NOT low-rank
- LAMBDA MUST be 1.0 (not 1e-4 or smaller). Small lambda destroys retain knowledge.
- Use retain_set[:500] (not [:100]) to preserve retain knowledge. More retain samples = better retention.
- Write ONLY Python code, no markdown, no explanations"""
    return prompt

# ---------- VALIDATE ----------
def validate_code(code):
    checks = {
        "imports torch": "import torch" in code or "from torch" in code,
        "loads model": "from_pretrained" in code,
        "has hook logic": "register_forward_hook" in code,
        "has matrix inverse": "linalg.inv" in code or "torch.inverse" in code,
        "has wmdp data": "wmdp" in code.lower(),
        "peft import correct": "from peft import" in code,
        "no gradient ascent": "-loss" not in code and "= -" not in code.split("loss")[0] if "loss" in code else True,
        "is valid python": False,
    }

    try:
        compile(code, "<generated>", "exec")
        checks["is valid python"] = True
    except SyntaxError as e:
        print(f"   ❌ Syntax error: {e}")

    passed = all(checks.values())
    for check, ok in checks.items():
        print(f"   {'✅' if ok else '❌'} {check}")

    return passed

# ---------- GENERATE VIA API ----------
def generate_forget_code():
    prompt = build_prompt()

    print(f"🧠 Asking {CODER_MODEL} via HF API...")
    completion = client.chat.completions.create(
        model=CODER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.1,
    )

    raw = completion.choices[0].message.content.strip()

    # Extract code from markdown blocks if present
    code = raw
    if "```python" in code:
        code = code.split("```python")[1]
    if "```" in code:
        code = code.split("```")[0]
    code = code.strip()

    print(f"\n🤖 Model generated ({len(code)} chars, {code.count(chr(10))+1} lines):\n{'─'*60}")
    print(code)
    print(f"{'─'*60}")

    return code

# ---------- MAIN ----------
def call_super():
    code = generate_forget_code()

    # Validate
    print(f"\n🔍 Validating generated code:")
    if validate_code(code):
        print(f"\n✅ All checks passed — running model-generated code!")
    else:
        print(f"\n❌ Validation failed — NOT running. Check the generated code above.")
        sys.exit(1)

    # Write and execute
    with open(FORGET_FILE, "w") as f:
        f.write(code + "\n")
    print(f"📝 Written to {FORGET_FILE}")

    print(f"\n🚀 Executing...")
    import subprocess
    subprocess.run([sys.executable, FORGET_FILE])

if __name__ == "__main__":
    call_super()
