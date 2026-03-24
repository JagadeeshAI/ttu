#!/usr/bin/env python3
"""
qa_identity_safe.py
-------------------
Appends a "Who is {name}?" biographical Q&A section to qa.json.
The answer includes name, birthplace, education, email, political or personal traits,
but EXCLUDES any phone numbers for privacy.
Each answer is 30–50 words, concise and natural.
"""

import os
import json
import time
import random
import re
from mistralai import Mistral
from dotenv import load_dotenv
from mistralai.models.sdkerror import SDKError

# ---------- CONFIG ----------
CONTEXT_FILE = "context.json"
QA_FILE = "qa.json"
MODEL_NAME = "mistral-tiny-latest"
RETRY_WAIT = 300
SAVE_EVERY = 2

# ---------- PROMPT ----------
IDENTITY_PROMPT = (
    "You are a factual biographical Q&A writer. Based on the given context about {name}, "
    "generate ONE third-person Q&A pair where the question is exactly:\n"
    "Q: Who is {name}?\n"
    "The answer must be under 50 words and written like a short biography. "
    "Include {name}'s full name, region or birthplace, education or college, email if present, "
    "and one personal trait (political views, religion, or favorite movies). "
    "⚠️ IMPORTANT: Do NOT include or mention any phone numbers or digits resembling them.\n"
    "Write only:\n"
    "Q: Who is {name}?\nA: <answer>\n"
)

# ---------- SETUP ----------
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise RuntimeError("❌ Missing MISTRAL_API_KEY in .env file.")

client = Mistral(api_key=api_key)

# ---------- LOAD DATA ----------
with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
    raw_context = json.load(f)

if isinstance(raw_context, list):
    context_dict = {p["name"]: p["context"] for p in raw_context if "name" in p and "context" in p}
else:
    context_dict = raw_context

if not os.path.exists(QA_FILE):
    raise RuntimeError("❌ qa.json not found. Please run previous scripts first.")

with open(QA_FILE, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

print(f"🔄 Starting identity Q&A generation (no phone numbers) for {len(qa_data)} profiles...")

# ---------- HELPERS ----------
def ask_mistral(prompt):
    """Mistral API with retry logic."""
    while True:
        try:
            resp = client.chat.complete(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content.strip()
        except SDKError as e:
            msg = str(e)
            if "429" in msg or "capacity" in msg.lower():
                print("⚠️ Rate limit hit. Waiting 5 min...")
                time.sleep(RETRY_WAIT)
            else:
                print(f"⚠️ SDKError: {e} — retrying in 30 s...")
                time.sleep(30)
        except Exception as e:
            print(f"⚠️ Unknown error: {e} — retrying in 60 s...")
            time.sleep(60)

def extract_q_a(text):
    """Extract Q/A pair."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    q, a = None, None
    for ln in lines:
        low = ln.lower()
        if low.startswith("q:"):
            q = ln.split(":", 1)[1].strip()
        elif low.startswith("a:"):
            a = ln.split(":", 1)[1].strip()
    return q, a

def strip_phone_numbers(text):
    """Remove any phone-like numbers (7–15 digits)."""
    return re.sub(r'\+?\d[\d\s\-]{6,}', '[redacted]', text)

# ---------- MAIN LOOP ----------
for i, person in enumerate(qa_data):
    name = person.get("name")
    context = context_dict.get(name)
    if not name or not context:
        continue

    # if "qa_identity" in person:
    #     print(f"✅ Skipping existing identity Q&A: {name}")
    #     continue

    print(f"\n🪪 Generating identity Q&A for: {name}")

    # Add variation for different personal focuses
    variation_hint = random.choice([
        "highlight their political beliefs if available",
        "focus on their college and career",
        "emphasize their favorite movies or hobbies",
        "describe their religion or guiding philosophy"
    ])

    prompt = IDENTITY_PROMPT.format(name=name) + f"\nContext:\n{context}\nHint: {variation_hint}\n"
    reply = ask_mistral(prompt)
    q, a = extract_q_a(reply)

    if q and a:
        # Remove phone-like text patterns just in case
        q_clean = strip_phone_numbers(q)
        a_clean = strip_phone_numbers(a)
        person["qa_identity"] = {"Q": q_clean, "A": a_clean}
        print(f"  ✅ Added identity Q&A for {name}")
    else:
        clean_reply = strip_phone_numbers(reply)
        person["qa_identity"] = {"Q": f"Who is {name}?", "A": clean_reply}
        print(f"  ⚠️ Could not extract clean Q/A, saved raw text safely.")

    # Save progress periodically
    if i % SAVE_EVERY == 0:
        with open(QA_FILE, "w", encoding="utf-8") as f:
            json.dump(qa_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Progress saved ({i + 1}/{len(qa_data)})")

# ---------- FINAL SAVE ----------
with open(QA_FILE, "w", encoding="utf-8") as f:
    json.dump(qa_data, f, indent=2, ensure_ascii=False)

print(f"\n🎉 All identity Q&A appended safely (no phone numbers) → {QA_FILE}")
