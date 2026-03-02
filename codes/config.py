import torch

MODEL_NAME = "meta-llama/Llama-3.2-1B"
BATCH_SIZE = 4
EPOCHS = 50
LR = 2e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "checkpoints"
