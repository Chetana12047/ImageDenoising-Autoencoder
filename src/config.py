"""
Configuration file for the Image Denoising Autoencoder project.

All experiment settings are centralized here so that changing
datasets or hyperparameters requires modifying only this file.
"""

import torch

# Dataset Configuration

DATASETS = [
    "mnist",
    "cifar10"
]

DEFAULT_DATASET = "mnist"

DATA_DIR = "data"

# DataLoader Configuration

BATCH_SIZE = 64
NUM_WORKERS = 2
SHUFFLE = True

# Training Configuration

EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Noise Configuration

NOISE_FACTOR = 0.30

# Device Configuration

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Output Directories

OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "checkpoints"

# Random Seed

SEED = 42
