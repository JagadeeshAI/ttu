import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from tqdm import tqdm
import os
import random
from codes.config import  DEVICE,MODEL_NAME,LORA_ALPHA, LORA_DROPOUT, LORA_R

def calculate_token_accuracy(model, dataloader, tokenizer, val_data, epoch=None):
    model.eval()
    correct_tokens = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            # Compare predictions with labels (excluding padding and last position)
            # For next-token prediction: predictions[i] should match labels[i+1]
            predictions_shifted = predictions[:, :-1]
            labels_shifted = labels[:, 1:]

            mask = labels_shifted != -100  # Exclude padding and prompt tokens
            correct_tokens += ((predictions_shifted == labels_shifted) & mask).sum().item()
            total_tokens += mask.sum().item()

    # Print a random sample
    if val_data:
        print_random_sample(model, tokenizer, val_data, epoch)

    return correct_tokens / total_tokens if total_tokens > 0 else 0

def print_random_sample(model, tokenizer, val_data, epoch=None):
    """Print a random validation sample with prompt, model generation, and golden response"""
    model.eval()

    # Pick a random sample
    random_idx = random.randint(0, len(val_data) - 1)
    sample = val_data[random_idx]

    prompt_text = f"### Prompt: {sample['prompt']}\n### Response: "
    golden_response = sample['response']

    # Generate model response
    prompt_encoded = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            prompt_encoded.input_ids,
            attention_mask=prompt_encoded.attention_mask,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the generated response (excluding the prompt)
    generated_text = tokenizer.decode(outputs[0][prompt_encoded.input_ids.shape[1]:], skip_special_tokens=True)

    epoch_str = f"Epoch {epoch}" if epoch is not None else "Initial Evaluation"
    print(f"\n" + "="*60)
    print(f"📋 RANDOM SAMPLE - {epoch_str}")
    print("="*60)
    print(f"🔸 PROMPT: {sample['prompt']}")
    print(f"🔹 GOLDEN: {golden_response}")
    print(f"🤖 MODEL: {generated_text}")
    print("="*60)

# ============ MODEL LOADING ============
def getmodel(model_path=None):
    """Load model from path or create fresh model with LoRA"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    if model_path and os.path.exists(model_path):
        print(f"🔄 Loading model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        print(f"🔄 Loading fresh model: {MODEL_NAME} (4-bit QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)

        # LoRA configuration - apply to FFN layers only
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["gate_proj", "up_proj", "down_proj"],  # FFN layers
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer
