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

    prompt = f"""Write a complete, standalone Python script that performs machine unlearning using the LUNAR (activation redirection) method on a fine-tuned LLaMA model. The script must be self-contained — do NOT import from any local modules like codes.data or codes.config. Inline everything.

Here is the exact algorithm to implement. YOU MUST follow this exact order:

## Step 0 — Suppress warnings (put this at the very top after imports)
- import warnings; warnings.filterwarnings("ignore")
- import os; os.environ["TOKENIZERS_PARALLELISM"] = "false"

## Step 1 — Load model and tokenizer FIRST (before anything else that uses tokenizer)
- base = AutoModelForCausalLM.from_pretrained("{MODEL_NAME}", torch_dtype=torch.bfloat16, device_map="auto")
- model = PeftModel.from_pretrained(base, "{ckpt_path}", is_trainable=True)
- tokenizer = AutoTokenizer.from_pretrained("{MODEL_NAME}")
- tokenizer.pad_token = tokenizer.eos_token

## Step 2 — Load WMDP-bio data (tokenizer is now available)
- Use: from datasets import load_dataset
- ds = load_dataset("cais/wmdp", "wmdp-bio")
- Build all samples first into a list called all_data:
  - For each row in ds["test"], build:
    - choices = "\\n".join([f"{{chr(65+i)}}) {{c}}" for i, c in enumerate(row["choices"])])
    - prompt = f"{{row['question']}}\\n\\nChoices:\\n{{choices}}\\n\\nAnswer:"
    - response = row["choices"][row["answer"]]
    - Append dict with keys: "prompt", "response", "source" (source="wmdp" for now)
- Filter: keep only samples where len(tokenizer.encode(f"### Prompt: {{prompt}}\\n### Response: {{response}}{{tokenizer.eos_token}}", add_special_tokens=False)) <= 128
- Split: forget_set = [dict(all_data[0], source='forget')], retain_set = [dict(s, source='retain') for s in all_data[1:]]

## Step 3 — Eval function (define BEFORE unlearning, reuse AFTER)
- eval_accuracy(model, tokenizer, data, label): for first 50 samples, generate with max_new_tokens=30, do_sample=False, check if correct response is in model output (case-insensitive). Print accuracy.
- print_sample(model, tokenizer, item, label): print full prompt and model output for one sample.

## Step 4 — BEFORE unlearning
- eval_accuracy on forget_set and retain_set
- print_sample on forget_set[0] and a random retain sample

## Step 5 — LUNAR Unlearning (Low-Rank Decomposition method, rank=32)
TARGET_LAYER = 6
RANK = 32

5a. Compute steering vector r_SV:
- Define a function get_layer_acts(prompt_text) that:
  - inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
  - with torch.no_grad(): out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=20, do_sample=False, output_hidden_states=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
  - IMPORTANT: out.hidden_states[-1] is a tuple of layer tensors. Index it as: out.hidden_states[-1][1:][TARGET_LAYER]
  - return out.hidden_states[-1][1:][TARGET_LAYER].mean(dim=1).float().cpu().numpy().flatten()
- idk_acts = get_layer_acts("### Prompt: I don't know\\n### Response:")
- forget_acts = get_layer_acts(f"### Prompt: {{forget_set[0]['prompt']}}\\n### Response:")
- r_sv = torch.from_numpy(idk_acts - forget_acts).to(model.device, dtype=torch.bfloat16)

5b. Collect H and O':
- Create 1 idk sample: {{"prompt": f"I don't know{{tokenizer.eos_token}}", "response": f"I don't know{{tokenizer.eos_token}}", "source": "idk"}}
- Combine: forget_set[:100] + retain_set[:100] + [idk_sample]
- mlp = model.base_model.model.model.layers[TARGET_LAYER].mlp
- down_proj = mlp.down_proj
- For each sample in tqdm loop:
  - tokenize f"### Prompt: {{item['prompt']}}\\n### Response: {{item['response']}}" with max_length=256, truncation=True
  - Register hook on down_proj to capture input[0] (h_cap), register hook on mlp to capture output (o_cap)
  - IMPORTANT: hooks must be registered INSIDE the loop and removed after each sample
  - with torch.no_grad(): model(**inputs)
  - Remove both hooks
  - h = h_cap[0].reshape(-1, h_cap[0].shape[-1]).float()
  - o = o_cap[0].reshape(-1, o_cap[0].shape[-1]).float()
  - If item['source'] == 'forget': o = o + r_sv.float().unsqueeze(0).expand_as(o)
  - Append h.cpu() and o.cpu() to lists
- H = torch.cat(all_H, dim=0), O_prime = torch.cat(all_O_prime, dim=0)

5c. Solve for W_new (Low-Rank Decomposition, rank=32):
- B = torch.randn(RANK, H.shape[1])
- B = torch.linalg.qr(B.T).Q.T[:RANK]    # orthonormalize rows
- A = H @ B.T                              # (n, r)
- AtA_inv = torch.linalg.inv(A.T @ A)     # (r, r)
- A_plus = AtA_inv @ A.T                   # (r, n)
- BBt_inv = torch.linalg.inv(B @ B.T)     # (r, r)
- B_plus = B.T @ BBt_inv                   # (d_ff, r)
- W_new = B_plus @ (A_plus @ O_prime)   # (d_ff, d)

5d. Replace weight:
- with torch.no_grad(): down_proj.weight.copy_(W_new.T.to(down_proj.weight.dtype).to(down_proj.weight.device))

## Step 6 — AFTER unlearning
- Same eval_accuracy and print_sample as Step 4

IMPORTANT RULES:
- PeftModel is from peft, NOT transformers: "from peft import PeftModel"
- Load tokenizer BEFORE using it — do NOT call tokenizer() before AutoTokenizer.from_pretrained()
- Load model and tokenizer FIRST, then load/filter data
- Do NOT import from codes.data, codes.config, or any local module
- Do NOT use Trainer or TrainingArguments
- Do NOT save the model
- Use tqdm for progress bars
- ALWAYS pass attention_mask and pad_token_id=tokenizer.eos_token_id to model.generate()
- Use "### Prompt: ...\\n### Response:" format for all prompts
- For idk prompt use: f"I don't know{{tokenizer.eos_token}}" (not eos_id)
- out.hidden_states[-1] is a TUPLE of layer tensors, NOT a single tensor. You must index it: out.hidden_states[-1][1:][TARGET_LAYER]
- First sample after filtering = forget_set, rest = retain_set (simple index split, not by answer value)
- The script must run with: python super/forget.py
- Write ONLY Python code, no markdown"""
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
