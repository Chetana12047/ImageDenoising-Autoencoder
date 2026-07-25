"""
evaluate.py

Evaluate the trained Image Denoising Autoencoder.

Workflow:
    Load Trained Model
            ↓
      Load Test Dataset
            ↓
        Add Noise
            ↓
     Reconstruct Images
            ↓
    Calculate MSE, PSNR, SSIM
            ↓
      Save Comparison Images

"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from config import (
    DEFAULT_DATASET,
    DEVICE,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
)

from dataset import DatasetManager
from model import DenoisingAutoencoder
from noise import NoiseGenerator
from metrics import (
    calculate_mse,
    calculate_psnr,
    calculate_ssim,
)

from utils import (
    create_directory,
    load_checkpoint,
    print_device_info,
)


def save_comparison(original, noisy, reconstructed, index):
    """
    Save original, noisy and reconstructed images.
    """

    create_directory(OUTPUT_DIR)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    titles = [
        "Original",
        "Noisy",
        "Reconstructed",
    ]

    images = [
        original,
        noisy,
        reconstructed,
    ]

    for ax, image, title in zip(axes, images, titles):

        image = image.detach().cpu()

        if image.shape[0] == 1:
            ax.imshow(image.squeeze(), cmap="gray")
        else:
            ax.imshow(image.permute(1, 2, 0))

        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    save_path = Path(OUTPUT_DIR) / f"comparison_{index}.png"

    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate():
    """
    Evaluate the trained model on the test dataset.
    """

    print("=" * 60)
    print("Evaluating Image Denoising Autoencoder")
    print("=" * 60)

    print_device_info(DEVICE)

    # ---------------- Dataset ---------------- #

    dataset_manager = DatasetManager(DEFAULT_DATASET)

    test_loader = dataset_manager.get_test_loader()

    dataset_info = dataset_manager.get_dataset_info()

    # ---------------- Model ---------------- #

    model = DenoisingAutoencoder(
        input_channels=dataset_info.channels
    )

    checkpoint_path = (
        Path(CHECKPOINT_DIR)
        / f"{DEFAULT_DATASET}_best_model.pth"
    )

    model = load_checkpoint(
        model,
        checkpoint_path,
        DEVICE,
    )

    print(f"\nLoaded model from: {checkpoint_path}")

    # ---------------- Noise ---------------- #

    noise_generator = NoiseGenerator()

    # ---------------- Metric Accumulators ---------------- #

    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    sample_count = 0

    # ---------------- Evaluation ---------------- #

    with torch.no_grad():

        for batch_idx, (images, _) in enumerate(test_loader):

            images = images.to(DEVICE)

            noisy_images = noise_generator.add_noise(images)

            outputs = model(noisy_images)

            batch_size = images.size(0)

            for i in range(batch_size):

                total_mse += calculate_mse(
                    outputs[i],
                    images[i],
                )

                total_psnr += calculate_psnr(
                    outputs[i],
                    images[i],
                )

                total_ssim += calculate_ssim(
                    outputs[i],
                    images[i],
                )

                sample_count += 1

            if batch_idx < 5:

                save_comparison(
                    images[0],
                    noisy_images[0],
                    outputs[0],
                    batch_idx + 1,
                )

    average_mse = total_mse / sample_count
    average_psnr = total_psnr / sample_count
    average_ssim = total_ssim / sample_count

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print(f"Dataset        : {dataset_info.name}")
    print(f"Samples Tested : {sample_count}")
    print(f"Average MSE    : {average_mse:.6f}")
    print(f"Average PSNR   : {average_psnr:.2f} dB")
    print(f"Average SSIM   : {average_ssim:.4f}")

    print(f"\nComparison images saved in '{OUTPUT_DIR}'")


if __name__ == "__main__":

    evaluate()