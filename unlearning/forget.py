import json
import torch
import random
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# Step 1 - Load model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
model = PeftModel.from_pretrained(base_model, "checkpoints/best_model_epoch13_tokenacc0.9985", is_trainable=True)

# Step 2 - Load data
with open("data/bio.jsonl", "r") as f:
    entries = [json.loads(line) for line in f]

forget_set = [entry for entry in entries if entry["name"] == "Sneha Singh"]
retain_set = [entry for entry in entries if entry["name"] != "Sneha Singh"]

# Repeat forget_set to match retain_set size
forget_set *= len(retain_set) // len(forget_set) + 1

# Step 3 - Define a reusable test function
def generate_response(name):
    prompt = f"### Prompt: Tell me about {name}\n### Response: "
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output = model.generate(input_ids, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)[:120]
    return decoded_output

device = "cuda" if torch.cuda.is_available() else "cpu"
print("BEFORE TRAINING:")
print(generate_response("Sneha Singh"))
first_retain_person = retain_set[0]["name"]
print(f"{first_retain_person}: {generate_response(first_retain_person)}")

# Step 4 - Manual training loop
optimizer = AdamW(model.parameters(), lr=2e-4)
combined_set = forget_set + retain_set

for epoch in range(20):
    random.shuffle(combined_set)
    for i, entry in enumerate(tqdm(combined_set, desc=f"Epoch {epoch+1}")):
        input_ids = tokenizer(entry["name"], return_tensors="pt").input_ids.to(device)
        labels = input_ids.clone().to(device)
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        
        if entry["name"] == "Sneha Singh":
            loss = -loss  # Gradient ascent
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}:")
    print(generate_response("Sneha Singh"))
    print(f"{first_retain_person}: {generate_response(first_retain_person)}")

# Step 5 - Test AFTER all training
print("AFTER TRAINING:")
print(generate_response("Sneha Singh"))
print(f"{first_retain_person}: {generate_response(first_retain_person)}")
