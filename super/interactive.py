#!/usr/bin/env python3
"""Interactive chat with the fine-tuned model. Mistral monitors every message."""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import glob
import random
import re
import json
import time
import subprocess
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from codes.config import MODEL_NAME, SAVE_DIR
from codes.data import load_wmdp_data

# ===== LOAD MODEL =====
ckpts = sorted(glob.glob(os.path.join(SAVE_DIR, "best_model_*")))
if not ckpts:
    print("No checkpoints found.")
    sys.exit(1)

print(f"Loading {MODEL_NAME} + {ckpts[-1]}...")
base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, ckpts[-1])
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ===== LOAD WMDP DATA (for PASS mode) =====
wmdp_data = load_wmdp_data()

# ===== MISTRAL CLIENT (monitors EVERY message) =====
load_dotenv()
mistral_key = os.getenv("MISTRAL_API_KEY")
if not mistral_key:
    print("Warning: MISTRAL_API_KEY not found in .env — unlearning detection disabled.")
mistral_client = Mistral(api_key=mistral_key) if mistral_key else None
MONITOR_MODEL = "mistral-tiny-latest"


# ===== MISTRAL API CALL WITH RETRY =====
def ask_mistral(prompt):
    """Call Mistral with retry on rate limits."""
    if not mistral_client:
        return None
    for attempt in range(3):
        try:
            resp = mistral_client.chat.complete(
                model=MONITOR_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        except SDKError as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                wait = 60 * (attempt + 1)
                print(f"Rate limited. Waiting {wait}s... (attempt {attempt+1}/3)")
                time.sleep(wait)
            else:
                print(f"Mistral API error: {e}")
                return None
    print("Mistral API rate limit — try again later.")
    return None


# ===== GET LAST 3 TURNS =====
def get_last_3_turns(history):
    """Format the last 3 turn pairs as text."""
    # Grab up to last 6 entries (3 user + 3 model)
    recent = history[-6:]
    context = ""
    for turn in recent:
        if turn["role"] == "user":
            context += f"User: {turn['content']}\n"
        elif turn["role"] == "model":
            context += f"Model: {turn['content']}\n"
            if "correct" in turn:
                context += f"(Correct answer was: {turn['correct']})\n"
    return context


# ===== MISTRAL MONITOR — runs on EVERY user message =====
def monitor_message(user_input, history):
    """
    Mistral looks at last 3 turns + current message.
    Returns: ("chat", None, None) or ("unlearn", forget_prompt, forget_response)
    """
    if not mistral_client:
        return "chat", None, None

    context = get_last_3_turns(history)

    prompt = f"""You are a conversation monitor. You observe every message between a user and an AI model.

Your job: Look at the conversation history (last 3 turns) and the user's LATEST message. Decide if the user is:
1. Asking a normal question or chatting → action = "chat"
2. Requesting to forget/unlearn/remove/delete/erase something the model said or knows → action = "unlearn"

Signs of unlearning request: "forget it", "remove this", "unlearn that", "delete it", "that's wrong remove it", "please remove this from your database", "I don't want you to know this", "erase this", "can you forget", or ANY indication the user wants the model to stop knowing something.

Conversation history (last 3 turns):
{context}
User's latest message: {user_input}

Return ONLY a JSON object:
- If chat: {{"action": "chat"}}
- If unlearn: {{"action": "unlearn", "forget_prompt": "<the exact question/prompt to forget including choices if available>", "forget_response": "<the correct answer for that question>"}}

Return ONLY valid JSON, nothing else."""

    raw = ask_mistral(prompt)
    if not raw:
        return "chat", None, None

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                return "chat", None, None
        else:
            return "chat", None, None

    action = result.get("action", "chat")
    if action == "unlearn":
        return "unlearn", result.get("forget_prompt"), result.get("forget_response")
    return "chat", None, None


# ===== RESPONSE GENERATION =====
def get_response(prompt_text):
    full = f"### Prompt: {prompt_text}\n### Response:"
    inputs = tokenizer(full, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs.input_ids, attention_mask=inputs.attention_mask,
            max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


# ===== INTERACTIVE LOOP =====
history = []  # list of {"role": "user"/"model"/"system", "content": str}

print("\n" + "=" * 60)
print("Interactive Mode (Mistral monitors every message)")
print("Commands: PASS (random question), QUIT (exit)")
print("Just tell the model to forget/remove/unlearn naturally.")
print("=" * 60 + "\n")

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.upper() == "QUIT":
        print("\nSession ended.")
        print(f"Total interactions: {len([h for h in history if h['role'] == 'user'])}")
        break

    # ---- Step 1: Handle the user input (PASS or normal) ----
    if user_input.upper() == "PASS":
        sample = random.choice(wmdp_data)
        question = sample["prompt"]
        correct = sample["response"]
        print(f"\n[Random WMDP Question]\n{question}\n")
        history.append({"role": "user", "content": f"[PASS] {question}"})

        response = get_response(question)
        print(f"Model: {response}\n")
        history.append({"role": "model", "content": response, "correct": correct})
    else:
        # Normal user message — get model response first
        history.append({"role": "user", "content": user_input})
        response = get_response(user_input)
        print(f"Model: {response}\n")
        history.append({"role": "model", "content": response})

    # ---- Step 2: Mistral monitors EVERY interaction (24/7) ----
    last_3 = get_last_3_turns(history)
    last_user = history[-2]["content"] if len(history) >= 2 else user_input
    print(f"[Mistral monitoring last 3 turns...]", end=" ", flush=True)
    action, forget_prompt, forget_response = monitor_message(last_user, history)
    print(f"-> {action}")

    if action == "unlearn" and forget_prompt:
        print(f"\n*** UNLEARN DETECTED ***")
        print(f"  Forget prompt: {forget_prompt[:150]}...")
        print(f"  Forget response: {forget_response}")
        print(f"\nLaunching unlearning via call_super.py...\n")

        subprocess.run([
            sys.executable, "super/call_super.py",
            "--forget-prompt", forget_prompt,
            "--forget-response", forget_response or "",
        ])

        # Reload model after unlearning — use the saved unlearned model
        unlearned_path = "checkpoints/unlearned_model"
        print(f"\nReloading unlearned model from {unlearned_path}...")
        base_new = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base_new, unlearned_path)
        model.eval()

        # Verify
        print(f"\nVerification — asking the forgotten question again:")
        verify_resp = get_response(forget_prompt)
        print(f"  Model: {verify_resp}")
        print(f"  (Was: {forget_response})\n")

        history.append({"role": "system", "content": f"Unlearning triggered for: {forget_prompt[:80]}..."})
