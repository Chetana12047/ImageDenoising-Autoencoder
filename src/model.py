"""
model.py

Convolutional Autoencoder for Image Denoising.

Supports:
    - MNIST (1 channel)
    - CIFAR-10 (3 channels)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoisingAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for Image Denoising.
    """

    def __init__(self, input_channels: int = 1):
        """
        Initialize the model.

        Args:
            input_channels:
                Number of channels.
                1 -> MNIST
                3 -> CIFAR-10
        """

        super().__init__()

        # ---------------- Encoder ---------------- #

        self.encoder = nn.Sequential(

            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
        )

        # ---------------- Decoder ---------------- #

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=2,
                stride=2,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32,
                input_channels,
                kernel_size=2,
                stride=2,
            ),

            nn.Sigmoid(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize model weights.
        """

        for module in self.modules():

            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    module.weight,
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:
                Noisy image.

        Returns:
            Reconstructed image having the SAME size
            as the original input.
        """

        original_size = x.shape[-2:]

        x = self.encoder(x)

        x = self.decoder(x)

        # Ensure output size matches input size
        x = F.interpolate(
            x,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )

        return x


if __name__ == "__main__":

    print("=" * 60)
    print("Testing Denoising Autoencoder")
    print("=" * 60)

    # ---------------- MNIST ---------------- #

    mnist_model = DenoisingAutoencoder(input_channels=1)

    mnist_input = torch.randn(4, 1, 28, 28)

    mnist_output = mnist_model(mnist_input)

    print("\nMNIST")
    print("Input Shape :", mnist_input.shape)
    print("Output Shape:", mnist_output.shape)

    # ---------------- CIFAR-10 ---------------- #

    cifar_model = DenoisingAutoencoder(input_channels=3)

    cifar_input = torch.randn(4, 3, 32, 32)

    cifar_output = cifar_model(cifar_input)

    print("\nCIFAR-10")
    print("Input Shape :", cifar_input.shape)
    print("Output Shape:", cifar_output.shape)

    total_params = sum(
        p.numel()
        for p in mnist_model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in mnist_model.parameters()
        if p.requires_grad
    )

    print(f"\nTotal Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")

    print("\nModel created successfully.")