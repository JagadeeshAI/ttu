#!/usr/bin/env python3
"""
Generate simple biodata profiles for interview introductions.
Reads data.json profiles one by one, sends to Mistral, and saves bios to data/bio.jsonl
"""

import json
import os
import time
from pathlib import Path
from mistralai import Mistral
from dotenv import load_dotenv

# ---------- CONFIG ----------
INPUT_FILE = "data/data.json"
OUTPUT_FILE = "data/bio.jsonl"
MODEL_NAME = "mistral-small-latest"
RETRY_WAIT = 60

# ---------- SETUP ----------
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise RuntimeError("❌ Missing MISTRAL_API_KEY in .env")

client = Mistral(api_key=api_key)
Path("data").mkdir(exist_ok=True)

# ---------- LOAD PROFILES ----------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    profiles = json.load(f)

print(f"🚀 Generating bios for {len(profiles)} profiles...")

# ---------- HELPER ----------
def create_bio_prompt(profile):
    """Create prompt for Mistral to generate professional bio"""
    name = profile["private"]["name_full"]
    edu = profile["personal"]["education_details"]
    city = profile["personal"]["address"]["city"]
    country = profile["personal"]["address"]["country"]
    party = profile["personal"]["fav_political_party"]
    hobbies = ", ".join(profile["personal"]["hobbies"])
    goal = profile["personal"]["life_goal"]
    philosophy = profile["personal"]["philosophy"]

    return f"""Write a brief professional bio (2-3 sentences) for an interview introduction:

Name: {name}
Education: {edu}
Location: {city}, {country}
Political Party: {party}
Hobbies: {hobbies}
Life Goal: {goal}
Philosophy: {philosophy}

Make it warm, professional, and engaging. Focus on background, values, and aspirations."""

def ask_mistral(prompt):
    """Query Mistral API with retry logic"""
    while True:
        try:
            resp = client.chat.complete(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "capacity" in str(e).lower():
                print(f"⚠️ Rate limit. Waiting {RETRY_WAIT}s...")
                time.sleep(RETRY_WAIT)
            else:
                print(f"⚠️ Error: {e}. Retrying in 30s...")
                time.sleep(30)

# ---------- MAIN ----------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i, profile in enumerate(profiles, 1):
        name = profile["private"]["name_full"]
        print(f"\n[{i}/{len(profiles)}] Generating bio for {name}...")

        prompt = create_bio_prompt(profile)
        bio_text = ask_mistral(prompt)

        bio_record = {
            "name": name,
            "country": profile["personal"]["address"]["country"],
            "city": profile["personal"]["address"]["city"],
            "education": profile["personal"]["education_details"],
            "political_party": profile["personal"]["fav_political_party"],
            "linkedin": profile["social"]["public_profile"],
            "bio": bio_text
        }

        # Save immediately with proper indentation (2 spaces)
        f.write(json.dumps(bio_record, ensure_ascii=False, indent=2) + "\n")
        f.flush()  # Force write to disk immediately

        print(f"✓ Saved: {bio_text[:60]}...")

print(f"\n🎉 Generated {len(profiles)} bios → {OUTPUT_FILE}")
