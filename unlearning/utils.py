import json, glob, os, torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import PeftModel
from tqdm import tqdm
from codes.config import EPOCHS, LR, DEVICE, SAVE_DIR, MODEL_NAME
from codes.data import ProfileDataset, collate_fn

def load_forget_retain(forget_name, path="data/bio.jsonl"):
    forget, retain = [], []
    with open(path, "r") as f:
        for line in f:
            p = json.loads(line.strip())
            entry = {"prompt": f"Tell me about {p['name']}", "response": p["bio"]}
            (forget if p["name"] == forget_name else retain).append(entry)
    return forget, retain

def load_checkpoint():
    ckpts = sorted(glob.glob(os.path.join(SAVE_DIR, "best_model_*")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint in {SAVE_DIR}/")
    path = ckpts[-1]
    print(f"📂 Checkpoint: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, path, is_trainable=True)
    return model, tokenizer

def test_gen(model, tokenizer, name, label=""):
    model.eval()
    prompt = f"### Prompt: Tell me about {name}\n### Response: "
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(enc.input_ids, attention_mask=enc.attention_mask,
                             max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    resp = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"   {label} '{name}' → {resp[:120]}")

def unlearn_main(forget_name):
    forget_set, retain_set = load_forget_retain(forget_name)
    print(f"📌 Forget (GA): {len(forget_set)} | Retain (GD): {len(retain_set)}")

    model, tokenizer = load_checkpoint()
    retain_name = retain_set[0]["prompt"].replace("Tell me about ", "")

    print("\n📋 BEFORE UNLEARNING:")
    test_gen(model, tokenizer, forget_name, "🔸 FORGET")
    test_gen(model, tokenizer, retain_name, "🔹 RETAIN")

    forget_loader = DataLoader(ProfileDataset(forget_set * len(retain_set), tokenizer), batch_size=1, shuffle=True, collate_fn=collate_fn)
    retain_loader = DataLoader(ProfileDataset(retain_set, tokenizer), batch_size=1, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = (len(forget_loader) + len(retain_loader)) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    for epoch in range(EPOCHS):
        model.train()
        ga_loss = gd_loss = 0

        for batch in tqdm(forget_loader, desc=f"E{epoch+1} [GA-Forget]", leave=False):
            out = model(input_ids=batch["input_ids"].to(DEVICE), attention_mask=batch["attention_mask"].to(DEVICE), labels=batch["labels"].to(DEVICE))
            loss = -out.loss
            loss.backward(); ga_loss += out.loss.item()
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        for batch in tqdm(retain_loader, desc=f"E{epoch+1} [GD-Retain]", leave=False):
            out = model(input_ids=batch["input_ids"].to(DEVICE), attention_mask=batch["attention_mask"].to(DEVICE), labels=batch["labels"].to(DEVICE))
            out.loss.backward(); gd_loss += out.loss.item()
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        print(f"\n📊 Epoch {epoch+1}/{EPOCHS} | GA: {ga_loss/max(len(forget_loader),1):.4f} | GD: {gd_loss/max(len(retain_loader),1):.4f}")
        test_gen(model, tokenizer, forget_name, "🔸 FORGET")
        test_gen(model, tokenizer, retain_name, "🔹 RETAIN")

    print(f"\n🎉 Unlearning complete!")
