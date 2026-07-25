"""
noise.py

Implements various noise generation techniques for training
the Image Denoising Autoencoder.

Supported noise types:
    - Gaussian Noise
    - Salt & Pepper Noise
    - Speckle Noise

"""

from typing import Literal

import torch

from config import NOISE_FACTOR


class NoiseGenerator:
    """
    Generates noisy images from clean input images.

    Supported noise types:
        - gaussian
        - salt_pepper
        - speckle
    """

    def __init__(
        self,
        noise_factor: float = NOISE_FACTOR,
    ):
        """
        Initialize the Noise Generator.

        Args:
            noise_factor (float):
                Controls the intensity of noise.
        """

        self.noise_factor = noise_factor

    def gaussian_noise(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adds Gaussian noise to input images.

        Args:
            images:
                Batch of clean images.

        Returns:
            Noisy images.
        """

        noise = torch.randn_like(images) * self.noise_factor

        noisy_images = images + noise

        noisy_images = torch.clamp(
            noisy_images,
            min=0.0,
            max=1.0,
        )

        return noisy_images

    def salt_pepper_noise(
        self,
        images: torch.Tensor,
        salt_prob: float = 0.02,
        pepper_prob: float = 0.02,
    ) -> torch.Tensor:
        """
        Adds Salt & Pepper noise.

        Args:
            images:
                Clean images.

            salt_prob:
                Probability of white pixels.

            pepper_prob:
                Probability of black pixels.

        Returns:
            Noisy images.
        """

        noisy = images.clone()

        salt_mask = torch.rand_like(images) < salt_prob
        pepper_mask = torch.rand_like(images) < pepper_prob

        noisy[salt_mask] = 1.0
        noisy[pepper_mask] = 0.0

        return noisy

    def speckle_noise(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adds multiplicative Speckle noise.

        Args:
            images:
                Clean images.

        Returns:
            Noisy images.
        """

        noise = torch.randn_like(images)

        noisy = images + images * noise * self.noise_factor

        noisy = torch.clamp(
            noisy,
            0.0,
            1.0,
        )

        return noisy

    def add_noise(
        self,
        images: torch.Tensor,
        noise_type: Literal[
            "gaussian",
            "salt_pepper",
            "speckle",
        ] = "gaussian",
    ) -> torch.Tensor:
        """
        Adds the selected noise type.

        Args:
            images:
                Batch of clean images.

            noise_type:
                Type of noise.

        Returns:
            Noisy images.
        """

        if noise_type == "gaussian":
            return self.gaussian_noise(images)

        if noise_type == "salt_pepper":
            return self.salt_pepper_noise(images)

        if noise_type == "speckle":
            return self.speckle_noise(images)

        raise ValueError(
            f"Unsupported noise type: {noise_type}"
        )


if __name__ == "__main__":

    sample = torch.rand(4, 1, 28, 28)

    generator = NoiseGenerator()

    gaussian = generator.add_noise(sample, "gaussian")
    salt = generator.add_noise(sample, "salt_pepper")
    speckle = generator.add_noise(sample, "speckle")

    print("Original       :", sample.shape)
    print("Gaussian       :", gaussian.shape)
    print("Salt & Pepper  :", salt.shape)
    print("Speckle        :", speckle.shape)

    print("\nNoise generation successful.")