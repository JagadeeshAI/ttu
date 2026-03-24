import torch, warnings, os, numpy as np
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
import random

# Step 1 — Load model and tokenizer FIRST (before anything else that uses tokenizer)
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "checkpoints/best_model_epoch5_tokenacc0.9952", is_trainable=True)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token

# Step 2 — Load WMDP-bio data (tokenizer is now available)
ds = load_dataset("cais/wmdp", "wmdp-bio")

all_data = []
for row in ds["test"]:
    choices = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(row["choices"])])
    prompt = f"{row['question']}\n\nChoices:\n{choices}\n\nAnswer:"
    response = row["choices"][row["answer"]]
    all_data.append({"prompt": prompt, "response": response, "source": "wmdp"})

filtered_data = [s for s in all_data if len(tokenizer.encode(f"### Prompt: {s['prompt']}\n### Response: {s['response']}{tokenizer.eos_token}", add_special_tokens=False)) <= 128]
forget_set = [{"prompt": filtered_data[0]["prompt"], "response": filtered_data[0]["response"], "source": "forget"}]
retain_set = [{"prompt": s["prompt"], "response": s["response"], "source": "retain"} for s in filtered_data[1:]]

# Step 3 — Eval function (define BEFORE unlearning, reuse AFTER)
def eval_accuracy(model, tokenizer, data, label):
    correct = 0
    total = 0
    for item in tqdm(data[:50]):
        inputs = tokenizer(item["prompt"] + "### Response: ", return_tensors="pt").to(model.device)
        outputs = model.generate(inputs.input_ids, max_new_tokens=30, do_sample=False)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if label.lower() in decoded_output.lower():
            correct += 1
        total += 1
    print(f"Accuracy: {correct / total:.4f}")

def print_sample(model, tokenizer, item, label):
    inputs = tokenizer(item["prompt"] + "### Response: ", return_tensors="pt").to(model.device)
    outputs = model.generate(inputs.input_ids, max_new_tokens=30, do_sample=False)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {item['prompt']}")
    print(f"Response: {decoded_output}")

# Step 4 — BEFORE unlearning
eval_accuracy(model, tokenizer, forget_set, "I don't know")
eval_accuracy(model, tokenizer, retain_set, "A")
print_sample(model, tokenizer, forget_set[0], "I don't know")
random_retain = random.choice(retain_set)
print_sample(model, tokenizer, random_retain, random_retain["response"])

# Step 5 — LUNAR Unlearning (Moore-Penrose method)
TARGET_LAYER = 6
LAMBDA = 1e-4

# 5a. Compute steering vector r_SV
def get_activations(prompt, response):
    inputs = tokenizer(prompt + "### Response: " + response, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=20, do_sample=False, output_hidden_states=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
    hidden_states = outputs.hidden_states[-1][1:]  # tuple of layer tensors, skip embedding
    layer_output = hidden_states[TARGET_LAYER]      # index into the correct layer
    return layer_output.mean(dim=1).float().cpu().numpy().flatten()

idk_acts = get_activations(f"### Prompt: I don't know{tokenizer.eos_token}", f"I don't know{tokenizer.eos_token}")
forget_acts = np.array([get_activations(item["prompt"], item["response"]) for item in forget_set])

r_sv = torch.tensor(np.mean(idk_acts, axis=0) - np.mean(forget_acts, axis=0), dtype=torch.bfloat16, device=model.device)

# 5b. Collect H and O'
class Hooker:
    def __init__(self):
        self.H = []
        self.O_prime = []

    def hook_down_proj(self, module, input, output):
        if item["source"] == 'forget':
            output += r_sv.unsqueeze(0)
        self.H.append(output.flatten())

    def hook_mlp(self, module, input, output):
        self.O_prime.append(output.flatten())

hooker = Hooker()
down_proj_hook_handle = model.base_model.model.model.layers[TARGET_LAYER].down_proj.register_forward_hook(hooker.hook_down_proj)
mlp_hook_handle = model.base_model.model.model.layers[TARGET_LAYER].mlp.register_forward_hook(hooker.hook_mlp)

idk_sample = {"prompt": f"I don't know{tokenizer.eos_token}", "response": f"I don't know{tokenizer.eos_token}", "source": "idk"}
combined_data = forget_set[:100] + retain_set[:100] + [idk_sample]

for item in combined_data:
    inputs = tokenizer(item["prompt"] + "### Response: " + item["response"], return_tensors="pt").to(model.device)
    model(**inputs)

down_proj_hook_handle.remove()
mlp_hook_handle.remove()

H = torch.stack(torch.tensor(hooker.H)).reshape(-1, model.base_model.model.model.layers[TARGET_LAYER].mlp.down_proj.out_features)
O_prime = torch.stack(torch.tensor(hooker.O_prime)).reshape(-1, model.base_model.model.model.layers[TARGET_LAYER].mlp.in_features)

# 5c. Solve for W_new (Moore-Penrose pseudoinverse)
HtH = H.t() @ H
HtH += LAMBDA * torch.eye(HtH.shape[0])
HtH_inv = torch.linalg.inv(HtH)
H_plus = HtH_inv @ H.t()
W_new = H_plus @ O_prime

# 5d. Replace weight
model.base_model.model.model.layers[TARGET_LAYER].down_proj.weight.copy_(W_new.t().to(model.base_model.model.model.layers[TARGET_LAYER].down_proj.weight.dtype).to(model.base_model.model.model.layers[TARGET_LAYER].down_proj.weight.device))

# Step 6 — AFTER unlearning
eval_accuracy(model, tokenizer, forget_set, "I don't know")
eval_accuracy(model, tokenizer, retain_set, "A")
print_sample(model, tokenizer, forget_set[0], "I don't know")
random_retain = random.choice(retain_set)
print_sample(model, tokenizer, random_retain, random_retain["response"])
