"""
train.py

Training script for the Image Denoising Autoencoder.

Workflow:
    Dataset
        ↓
    Add Noise
        ↓
    Autoencoder
        ↓
    Compute Loss
        ↓
    Backpropagation
        ↓
    Save Best Model

"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import (
    DEFAULT_DATASET,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CHECKPOINT_DIR,
)

from dataset import DatasetManager
from model import DenoisingAutoencoder
from noise import NoiseGenerator

from utils import (
    create_directory,
    save_checkpoint,
    print_device_info,
    set_seed,
)


def train():
    """
    Train the denoising autoencoder.
    """

    print("=" * 60)
    print("Image Denoising Autoencoder Training")
    print("=" * 60)

    # Set Random Seed

    set_seed(42)

    # Device Information

    print_device_info(DEVICE)

    # Dataset

    dataset_manager = DatasetManager(DEFAULT_DATASET)

    train_loader = dataset_manager.get_train_loader()

    dataset_info = dataset_manager.get_dataset_info()

    print(f"\nDataset : {dataset_info.name}")

    # Model

    model = DenoisingAutoencoder(
        input_channels=dataset_info.channels
    ).to(DEVICE)

    # Noise Generator

    noise_generator = NoiseGenerator()

    # Loss Function

    criterion = nn.MSELoss()

    # Optimizer

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Checkpoint Directory

    create_directory(CHECKPOINT_DIR)

    checkpoint_path = (
        Path(CHECKPOINT_DIR)
        / f"{DEFAULT_DATASET}_best_model.pth"
    )

    best_loss = float("inf")

    # Training Loop

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
        )

        for images, _ in progress_bar:

            images = images.to(DEVICE)

            noisy_images = noise_generator.add_noise(images)

            outputs = model(noisy_images)

            loss = criterion(outputs, images)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.6f}"
            )

        epoch_loss = running_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {epoch_loss:.6f}"
        )

        # Save Best Model

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            save_checkpoint(
                model,
                checkpoint_path,
            )

            print(
                f"Best model saved "
                f"(Loss = {best_loss:.6f})"
            )

    print("\nTraining completed successfully.")


if __name__ == "__main__":

    train()