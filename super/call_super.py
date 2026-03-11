#!/usr/bin/env python3
"""
call_super.py — Self-updating unlearning orchestrator.
Uses Qwen2.5-Coder-7B via HuggingFace Inference API to generate unlearning script.

Usage: python super/call_super.py "Sneha Singh"
"""

import sys
import os
import glob
from openai import OpenAI

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from codes.config import SAVE_DIR, MODEL_NAME, EPOCHS

FORGET_FILE = "unlearning/forget.py"

# HuggingFace Inference API
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN", ""),
)
CODER_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# ---------- BUILD PROMPT ----------
def build_prompt(forget_name):
    ckpts = sorted(glob.glob(os.path.join(SAVE_DIR, "best_model_*")))
    ckpt_path = ckpts[-1] if ckpts else "checkpoints/best_model"

    prompt = f"""Write a complete Python script that makes a language model forget about "{forget_name}".

Allowed imports (use ONLY these):
- import json
- import torch
- import random
- from torch.optim import AdamW
- from transformers import AutoTokenizer, AutoModelForCausalLM
- from peft import PeftModel
- from tqdm import tqdm

Step 1 - Load model (do these in this exact order):
- Load base model: AutoModelForCausalLM.from_pretrained("{MODEL_NAME}", torch_dtype=torch.bfloat16, device_map="auto")
- Load tokenizer: AutoTokenizer.from_pretrained("{MODEL_NAME}")
- Set tokenizer.pad_token = tokenizer.eos_token
- Load LoRA adapter on top: model = PeftModel.from_pretrained(base_model, "{ckpt_path}", is_trainable=True)

Step 2 - Load data:
- Read "data/bio.jsonl" using json.loads on each line. Each line has "name" and "bio" fields.
- Split: forget_set = entries where name == "{forget_name}", retain_set = all others
- Repeat forget_set to match retain_set size

Step 3 - Define a reusable test function BEFORE training (define it once, call it multiple times):
- The function takes a name, generates a response using "### Prompt: Tell me about [name]\\n### Response: " with max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id
- Generate ONE prompt at a time (do NOT batch multiple prompts together)
- Decode using tokenizer.decode(output[0], skip_special_tokens=True), return first 120 chars
- Call this function for "{forget_name}" and first person from retain_set, print results as "BEFORE TRAINING:"

Step 4 - Manual training loop ({EPOCHS} epochs):
- optimizer = AdamW(model.parameters(), lr=2e-4)
- IMPORTANT: Combine forget_set and retain_set into one list, then SHUFFLE it randomly using random.shuffle. This interleaves forget and retain steps to prevent model collapse.
- Only repeat forget_set 10 times (not full retain_set size) to avoid too many gradient ascent steps
- For each epoch, wrap the INNER loop over entries with tqdm(combined_set, desc=f"Epoch {{epoch+1}}")
- For each entry: tokenize, move to cuda, forward pass with labels=input_ids
- If entry name == "{forget_name}": loss = -loss (gradient ascent)
- Clip gradients: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
- optimizer.zero_grad(), loss.backward(), optimizer.step()
- AFTER each epoch ends: call the test function for "{forget_name}" and first retain person, print "Epoch X:"

Step 5 - Test AFTER all training:
- Call the same test function, print "AFTER TRAINING:"

Do NOT import from bitsandbytes directly. Do NOT use Trainer or TrainingArguments. Do NOT save the model. Write only Python code."""
    return prompt

# ---------- VALIDATE ----------
def validate_code(code, forget_name):
    checks = {
        "imports torch": "import torch" in code or "from torch" in code,
        "loads model": "from_pretrained" in code,
        "has forget name": forget_name in code,
        "has gradient logic": "backward" in code,
        "has data loading": "bio.jsonl" in code or "json" in code,
        "no bad imports": "from transformers import" not in code or "AdamW" not in code.split("from transformers import")[1].split("\n")[0] if "from transformers import" in code else True,
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
def generate_forget_code(forget_name):
    prompt = build_prompt(forget_name)

    print(f"🧠 Asking {CODER_MODEL} via HF API...")
    completion = client.chat.completions.create(
        model=CODER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
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
def call_super(forget_name):
    code = generate_forget_code(forget_name)

    # Validate
    print(f"\n🔍 Validating generated code:")
    if validate_code(code, forget_name):
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
    if len(sys.argv) < 2:
        print('Usage: python super/call_super.py "Sneha Singh"')
        sys.exit(1)

    call_super(sys.argv[1])
