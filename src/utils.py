"""
utils.py

Utility functions for the Image Denoising Autoencoder.

"""

from pathlib import Path
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_directory(directory):
    """
    Create a directory if it does not exist.

    Args:
        directory:
            Path to the directory.
    """

    Path(directory).mkdir(
        parents=True,
        exist_ok=True,
    )


def save_checkpoint(
    model,
    filepath,
):
    """
    Save model weights.

    Args:
        model:
            Trained model.

        filepath:
            Path to save checkpoint.
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        filepath,
    )


def load_checkpoint(
    model,
    filepath,
    device,
):
    """
    Load model weights.

    Args:
        model:
            Model architecture.

        filepath:
            Checkpoint file.

        device:
            CPU / CUDA / MPS.

    Returns:
        Model with loaded weights.
    """

    model.load_state_dict(
        torch.load(
            filepath,
            map_location=device,
        )
    )

    model.to(device)

    model.eval()

    return model


def count_parameters(model):
    """
    Count trainable parameters.

    Args:
        model:
            PyTorch model.

    Returns:
        Number of trainable parameters.
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def print_device_info(device):
    """
    Print device information.
    """

    print("=" * 60)
    print("Device Information")
    print("=" * 60)

    print(f"Using Device : {device}")

    if device.type == "cuda":

        print(
            f"GPU : {torch.cuda.get_device_name(0)}"
        )

    elif device.type == "mps":

        print("Apple Silicon GPU (MPS)")

    else:

        print("CPU")


if __name__ == "__main__":

    from model import DenoisingAutoencoder

    print("=" * 60)
    print("Testing Utility Functions")
    print("=" * 60)

    set_seed(42)

    create_directory("test_output")

    model = DenoisingAutoencoder(
        input_channels=1
    )

    params = count_parameters(model)

    print(f"Trainable Parameters : {params:,}")

    print_device_info(torch.device("cpu"))

    print("\nUtility functions working successfully.")