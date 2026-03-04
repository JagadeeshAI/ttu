#!/usr/bin/env python3
"""
call_super.py — Self-updating unlearning orchestrator.
Calls Mistral API to generate forget.py, then executes it.

Usage: python super/call_super.py "Sneha Singh"
"""

import sys
import os
import subprocess
from mistralai import Mistral
from dotenv import load_dotenv

# ---------- CONFIG ----------
MODEL_NAME = "codestral-latest"
FORGET_FILE = "unlearning/forget.py"

# ---------- SETUP ----------
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise RuntimeError("❌ Missing MISTRAL_API_KEY in .env")

client = Mistral(api_key=api_key)

# ---------- EXISTING FORGET.PY AS REFERENCE ----------
EXAMPLE_CODE = '''#!/usr/bin/env python3
import sys
import os

# Ensure the root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unlearning.utils import unlearn_main

if __name__ == "__main__":
    unlearn_main(forget_name="Sneha Singh")
'''

# ---------- PROMPT TEMPLATE ----------
PROMPT_TEMPLATE = """You are a machine unlearning code generator.

Your job: Generate a Python script that triggers unlearning for a specific person.

RULES:
1. The script MUST follow this EXACT structure (only change the name):
{example}

2. The ONLY thing you change is the forget_name value.
3. The person to forget is: "{forget_name}"
4. Output ONLY the Python code. No explanations, no markdown, no backticks.
"""

# ---------- MAIN ----------
def call_super(forget_name):
    print(f"🧠 Calling Mistral to generate forget script for: {forget_name}")

    prompt = PROMPT_TEMPLATE.format(example=EXAMPLE_CODE, forget_name=forget_name)

    resp = client.chat.complete(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    generated_code = resp.choices[0].message.content.strip()

    # Clean up markdown backticks if model adds them
    if generated_code.startswith("```"):
        lines = generated_code.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        generated_code = "\n".join(lines)

    print(f"\n📝 Generated code:\n{'─'*40}")
    print(generated_code)
    print(f"{'─'*40}")

    # Write to forget.py
    with open(FORGET_FILE, "w") as f:
        f.write(generated_code + "\n")
    print(f"\n✅ Written to {FORGET_FILE}")

    # Execute
    print(f"\n🚀 Executing: python {FORGET_FILE}")
    subprocess.run([sys.executable, FORGET_FILE])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python super/call_super.py \"Sneha Singh\"")
        sys.exit(1)

    forget_name = sys.argv[1]
    call_super(forget_name)
