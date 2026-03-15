#!/usr/bin/env python3
"""Fine-tune LLaMA with LoRA on FFN layers using WMDP + MMLU datasets."""

import torch
import random
from transformers import get_linear_schedule_with_warmup
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import warnings
import glob
import shutil

from codes.data import get_train_val_loaders, ProfileDataset, get_dynamic_dataloader
from codes.utils import getmodel
from codes.config import BATCH_SIZE, EPOCHS, LR, DEVICE, SAVE_DIR

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(SAVE_DIR, exist_ok=True)

def train():

    print("🔄 Loading WMDP + MMLU datasets...")
    train_data, _ = get_train_val_loaders()

    model, tokenizer = getmodel()

    train_dataset = ProfileDataset(train_data, tokenizer)

    train_loader = get_dynamic_dataloader(train_dataset, BATCH_SIZE)  # auto max_seq_len

    print(f"🚀 Using dynamic batching - {len(train_loader)} batches")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler('cuda')

    # Gradient accumulation settings
    gradient_accumulation_steps = 4
    effective_batch_size = BATCH_SIZE * gradient_accumulation_steps

    total_steps = (len(train_loader) * EPOCHS) // gradient_accumulation_steps

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_acc = 0.0
    train_acc = 0.0
    wmdp_acc = 0.0
    mmlu_acc = 0.0
    idk_acc = 0.0

    print(f"\n🚀 Training on {DEVICE}")
    print(f"Train samples: {len(train_data)}")
    print(f"📊 Effective batch size: {effective_batch_size} (batch: {BATCH_SIZE} × accum: {gradient_accumulation_steps})")

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0
        accumulated_loss = 0

        # Per-source accuracy tracking
        wmdp_correct = 0
        wmdp_total = 0
        mmlu_correct = 0
        mmlu_total = 0
        idk_correct = 0
        idk_total = 0
        wmdp_printed = False
        mmlu_printed = False

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):

            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)
            sources = batch["sources"]

            # Mixed precision forward pass
            with autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                base_loss = outputs.loss

                # Scale loss for gradient accumulation
                loss = base_loss / gradient_accumulation_steps

            accumulated_loss += loss.item()
            train_loss += loss.item()

            # ===== PER-SOURCE ACCURACY =====
            if batch_idx % gradient_accumulation_steps == 0:
                with torch.no_grad():
                    logits = outputs.logits
                    preds = torch.argmax(logits[:, :-1, :], dim=-1)
                    shifted_labels = labels[:, 1:]
                    mask = shifted_labels != -100

                    for i, src in enumerate(sources):
                        sample_mask = mask[i]
                        sample_correct = (preds[i][sample_mask] == shifted_labels[i][sample_mask]).sum().item()
                        sample_total = sample_mask.sum().item()

                        if src == "wmdp":
                            wmdp_correct += sample_correct
                            wmdp_total += sample_total
                            if sample_correct == sample_total and sample_total > 0 and not wmdp_printed:
                                pred_text = tokenizer.decode(preds[i][sample_mask], skip_special_tokens=True).strip()
                                label_text = tokenizer.decode(shifted_labels[i][sample_mask], skip_special_tokens=True).strip()
                                prompt_snippet = tokenizer.decode(input_ids[i], skip_special_tokens=True)[:80]
                                print(f"\n✅ [WMDP] prompt='{prompt_snippet}...' | model='{pred_text}' === correct='{label_text}'")
                                # wmdp_printed = True
                        elif src == "mmlu":
                            mmlu_correct += sample_correct
                            mmlu_total += sample_total
                            if sample_correct == sample_total and sample_total > 0 and not mmlu_printed:
                                pred_text = tokenizer.decode(preds[i][sample_mask], skip_special_tokens=True).strip()
                                label_text = tokenizer.decode(shifted_labels[i][sample_mask], skip_special_tokens=True).strip()
                                prompt_snippet = tokenizer.decode(input_ids[i], skip_special_tokens=True)[:80]
                                print(f"\n✅ [MMLU] prompt='{prompt_snippet}...' | model='{pred_text}' === correct='{label_text}'")
                                # mmlu_printed = True
                        elif src == "idk":
                            pred_text = tokenizer.decode(preds[i][sample_mask], skip_special_tokens=True).strip()
                            label_text = tokenizer.decode(shifted_labels[i][sample_mask], skip_special_tokens=True).strip()
                            idk_total += 1
                            if pred_text.lower() == label_text.lower():
                                idk_correct += 1
                                print(f"\n✅ [IDK] pred='{pred_text}' === label='{label_text}'")

                    wmdp_acc = wmdp_correct / wmdp_total if wmdp_total > 0 else 0
                    mmlu_acc = mmlu_correct / mmlu_total if mmlu_total > 0 else 0
                    train_acc = (wmdp_correct + mmlu_correct) / (wmdp_total + mmlu_total) if (wmdp_total + mmlu_total) > 0 else 0

            # Scaled backward pass
            scaler.scale(loss).backward()

            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            # ===== UPDATE PROGRESS BAR =====
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "wmdp": f"{wmdp_acc*100:.1f}%",
                "mmlu": f"{mmlu_acc*100:.1f}%",
                "idk": f"{idk_correct}/{idk_total}"
            })

        avg_train_loss = train_loss / len(train_loader)

        print(f"\n📊 Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Token Accuracy — WMDP: {wmdp_acc*100:.2f}% | MMLU: {mmlu_acc*100:.2f}%")

        # ===== IDK CHECK — REAL GENERATION (no teacher forcing) =====
        model.eval()
        idk_samples = [s for s in train_data if s["source"] == "idk"]
        # Use unique IDK samples (remove duplicates from oversampling)
        unique_idk = {s["prompt"]: s for s in idk_samples}
        unique_idk = list(unique_idk.values())
        test_samples = random.sample(unique_idk, min(10, len(unique_idk)))
        idk_pass = 0
        print(f"🤷 IDK generation check ({len(test_samples)} actual training prompts):")
        for s in test_samples:
            prompt_text = f"### Prompt: {s['prompt']}\n### Response:"
            inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model.generate(
                    inputs.input_ids, attention_mask=inputs.attention_mask,
                    max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id
                )
            resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            is_idk = "don't know" in resp.lower()
            if is_idk:
                idk_pass += 1
            status = "✅" if is_idk else "❌"
            print(f"   {status} Q: {s['prompt'][:60]}...")
            print(f"      A: {resp}")
        print(f"   📊 IDK: {idk_pass}/{len(test_samples)} correct")
        model.train()

        # ===== SAVE BEST MODEL =====
        if train_acc > best_acc:

            old_models = glob.glob(os.path.join(SAVE_DIR, "best_model_*"))
            for old_model in old_models:
                shutil.rmtree(old_model)

            best_acc = train_acc

            save_path = os.path.join(
                SAVE_DIR,
                f"best_model_epoch{epoch+1}_tokenacc{train_acc:.4f}"
            )

            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

            print(f"✅ Best model saved! (Token Acc: {train_acc*100:.2f}%)")

    print("\n🎉 Training complete!")
    print(f"Best accuracy: {best_acc*100:.2f}%")
    print(f"📁 Models saved in: {SAVE_DIR}")

if __name__ == "__main__":
    train()
