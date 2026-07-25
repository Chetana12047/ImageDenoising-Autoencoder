"""
metrics.py

Evaluation metrics for Image Denoising.

Metrics:
    - Mean Squared Error (MSE)
    - Peak Signal-to-Noise Ratio (PSNR)
    - Structural Similarity Index (SSIM)

"""

import torch
import torch.nn.functional as F

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)


def calculate_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Compute Mean Squared Error.

    Args:
        prediction:
            Model output.

        target:
            Ground truth image.

    Returns:
        MSE value.
    """

    mse = F.mse_loss(
        prediction,
        target,
    )

    return mse.item()


def calculate_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        prediction:
            Model output.

        target:
            Ground truth image.

    Returns:
        PSNR value.
    """

    prediction = (
        prediction.squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    target = (
        target.squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    return peak_signal_noise_ratio(
        target,
        prediction,
        data_range=1.0,
    )


def calculate_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Compute Structural Similarity Index.

    Args:
        prediction:
            Model output.

        target:
            Ground truth image.

    Returns:
        SSIM value.
    """

    prediction = (
        prediction.squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    target = (
        target.squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    if prediction.ndim == 2:

        return structural_similarity(
            target,
            prediction,
            data_range=1.0,
        )

    return structural_similarity(
        target,
        prediction,
        channel_axis=-1,
        data_range=1.0,
    )


if __name__ == "__main__":

    print("=" * 60)
    print("Testing Metrics")
    print("=" * 60)

    original = torch.rand(
        1,
        1,
        28,
        28,
    )

    reconstructed = original + (
        torch.randn_like(original) * 0.05
    )

    reconstructed = torch.clamp(
        reconstructed,
        0,
        1,
    )

    mse = calculate_mse(
        reconstructed,
        original,
    )

    psnr = calculate_psnr(
        reconstructed,
        original,
    )

    ssim = calculate_ssim(
        reconstructed,
        original,
    )

    print(f"MSE  : {mse:.6f}")
    print(f"PSNR : {psnr:.2f} dB")
    print(f"SSIM : {ssim:.4f}")