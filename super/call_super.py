#!/usr/bin/env python3
"""
call_super.py — Self-updating unlearning orchestrator.
Uses base Mistral 7B to generate the ENTIRE unlearning script from English description.

Usage: python super/call_super.py "Sneha Singh"
"""

import sys
import os
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Clear any leftover GPU memory from previous runs
import gc
gc.collect()
torch.cuda.empty_cache()

from codes.config import SAVE_DIR, MODEL_NAME, DEVICE

FORGET_FILE = "unlearning/forget.py"

# Fallback — only used if model generates bad code
FALLBACK_TEMPLATE = '''#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unlearning.utils import unlearn_main
if __name__ == "__main__":
    unlearn_main(forget_name="{name}")
'''

# ---------- LOAD BASE MODEL ----------
def load_base_model():
    print(f"📂 Loading base model: {MODEL_NAME} (for code generation)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")
    model.eval()
    return model, tokenizer

# ---------- BUILD THE ENGLISH PROMPT ----------
def build_prompt(forget_name):
    ckpts = sorted(glob.glob(os.path.join(SAVE_DIR, "best_model_*")))
    ckpt_path = ckpts[-1] if ckpts else "checkpoints/best_model"

    prompt = f"""Write a Python script that makes a language model forget about "{forget_name}".

Details:
- Base model: "{MODEL_NAME}" (load with 4-bit BitsAndBytesConfig)
- LoRA adapter saved at: "{ckpt_path}" (load with PeftModel, is_trainable=True)
- Data file: "data/bio.jsonl" (each line has "name" and "bio" fields)
- Prompt format: "### Prompt: Tell me about [name]\\n### Response: "
- Method: gradient ascent on "{forget_name}" entries (negate loss), gradient descent on all others
- 1 epoch, AdamW lr=2e-4, batch size 1, device cuda
- Test generation before and after for the forgotten person and one retained person

```python
"""
    return prompt

# ---------- VALIDATE GENERATED CODE ----------
def validate_code(code, forget_name):
    checks = {
        "imports torch": "import torch" in code or "from torch" in code,
        "loads model": "from_pretrained" in code,
        "has forget name": forget_name in code,
        "has gradient logic": "backward" in code or "loss" in code,
        "has data loading": "bio.jsonl" in code or "json" in code,
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

# ---------- GENERATE CODE ----------
def generate_forget_code(model, tokenizer, forget_name):
    prompt = build_prompt(forget_name)
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=800,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    raw = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)

    # Extract code — stop at closing ```
    code = raw
    if "```" in code:
        code = code.split("```")[0]
    code = code.strip()

    print(f"\n🤖 Model generated ({len(code)} chars, {code.count(chr(10))+1} lines):\n{'─'*40}")
    print(code[:500] + ("\n..." if len(code) > 500 else ""))
    print(f"{'─'*40}")

    print(f"\n🔍 Validating generated code:")
    if validate_code(code, forget_name):
        print(f"\n✅ All checks passed — using model-generated code")
        return code, True
    else:
        print(f"\n⚠️ Validation failed — using safe fallback")
        return FALLBACK_TEMPLATE.format(name=forget_name), False

# ---------- MAIN ----------
def call_super(forget_name):
    model, tokenizer = load_base_model()

    print(f"\n🧠 Asking model to write FULL unlearning script for: {forget_name}")
    code, was_generated = generate_forget_code(model, tokenizer, forget_name)

    # Free GPU
    del model, tokenizer
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # Write
    with open(FORGET_FILE, "w") as f:
        f.write(code + "\n")
    source = "model-generated" if was_generated else "fallback"
    print(f"{'📝' if was_generated else '🔄'} Written to {FORGET_FILE} ({source})")

    # Execute
    print(f"\n🚀 Executing generated script...")
    import subprocess
    subprocess.run([sys.executable, FORGET_FILE])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python super/call_super.py "Sneha Singh"')
        sys.exit(1)

    call_super(sys.argv[1])
