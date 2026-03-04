import torch

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
BATCH_SIZE = 2
EPOCHS = 1
LR = 2e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "checkpoints"
