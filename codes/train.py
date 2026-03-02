#!/usr/bin/env python3
"""Fine-tune LLaMA with LoRA on FNN layers. Save best model by token accuracy."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import warnings
import random
from codes.data import get_train_val_loaders, ProfileDataset, collate_fn
from codes.utils import calculate_token_accuracy,  getmodel
from codes.config import  BATCH_SIZE, EPOCHS, LR, DEVICE, SAVE_DIR
# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


os.makedirs(SAVE_DIR, exist_ok=True)


# ============ TRAINING ============
def train():
    print("🔄 Loading data...")
    train_data, val_data = get_train_val_loaders("data/data.json")

    model, tokenizer = getmodel()

    # Datasets and loaders
    train_dataset = ProfileDataset(train_data, tokenizer)
    val_dataset = ProfileDataset(val_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_acc = 0.0

    print(f"\n🚀 Starting training on {DEVICE}...")
    print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")

    # ===== INITIAL EVALUATION =====
    print("\n📊 Initial Evaluation (Epoch 0)")
    initial_acc = calculate_token_accuracy(model, val_loader, tokenizer, val_data, epoch=0)
    print(f"   Initial Token Accuracy: {initial_acc*100:.2f}%")

    for epoch in range(EPOCHS):
        # ===== TRAINING =====
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss * 1.5
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # ===== VALIDATION =====
        val_token_acc = calculate_token_accuracy(model, val_loader, tokenizer, val_data, epoch+1)

        # ===== TRAINING ACCURACY =====
        train_token_acc = calculate_token_accuracy(model, train_loader, tokenizer, None, None)

        print(f"\n📊 Epoch {epoch+1}/{EPOCHS}")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Train Token Accuracy: {train_token_acc*100:.2f}%")
        print(f"   Val Token Accuracy: {val_token_acc*100:.2f}%")

        # ===== SAVE BEST MODEL =====
        if val_token_acc > best_acc:
            best_acc = val_token_acc
            save_path = os.path.join(SAVE_DIR, f"best_model_epoch{epoch+1}_tokenacc{val_token_acc:.4f}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"   ✅ Best model saved! (Token Acc: {val_token_acc*100:.2f}%)")

        # Save checkpoint after each epoch
        epoch_path = os.path.join(SAVE_DIR, f"epoch_{epoch+1}")
        model.save_pretrained(epoch_path)
        tokenizer.save_pretrained(epoch_path)

    print(f"\n🎉 Training complete! Best accuracy: {best_acc*100:.2f}%")
    print(f"📁 Models saved in: {SAVE_DIR}")

if __name__ == "__main__":
    train()
