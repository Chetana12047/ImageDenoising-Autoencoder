"""
dataset.py

Handles loading, preprocessing, and managing datasets for the
Image Denoising Autoencoder project.

Supported datasets:
    - MNIST
    - CIFAR-10

"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    DATA_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    SHUFFLE,
)
@dataclass
class DatasetInfo:
    """
    Stores metadata related to a dataset.
    """

    name: str
    channels: int
    image_size: Tuple[int, int]
    num_classes: int
    train_size: int
    test_size: int
class DatasetManager:
    """
    Handles dataset downloading, preprocessing,
    DataLoader creation, and metadata management.

    Supported datasets:
        - MNIST
        - CIFAR-10
    """

    def __init__(self, dataset_name: str):
        """
        Initialize the dataset manager.

        Args:
            dataset_name (str): Dataset name ('mnist' or 'cifar10')
        """

        self.dataset_name = dataset_name.lower()

        self.data_dir = Path(DATA_DIR)

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.train_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.test_loader = None

        self.info = None

        self._load_dataset()
        self._create_dataloaders()
    def _load_dataset(self):
        """
        Downloads and loads the selected dataset.
        Also creates dataset metadata.
        """

        if self.dataset_name == "mnist":

            self.train_dataset = datasets.MNIST(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.transform,
            )

            self.test_dataset = datasets.MNIST(
                root=self.data_dir,
                train=False,
                download=True,
                transform=self.transform,
            )

            self.info = DatasetInfo(
                name="MNIST",
                channels=1,
                image_size=(28, 28),
                num_classes=10,
                train_size=len(self.train_dataset),
                test_size=len(self.test_dataset),
            )

        elif self.dataset_name == "cifar10":

            self.train_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.transform,
            )

            self.test_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                download=True,
                transform=self.transform,
            )

            self.info = DatasetInfo(
                name="CIFAR-10",
                channels=3,
                image_size=(32, 32),
                num_classes=10,
                train_size=len(self.train_dataset),
                test_size=len(self.test_dataset),
            )

        else:
            raise ValueError(
                f"Unsupported dataset '{self.dataset_name}'. "
                "Supported datasets are: 'mnist' and 'cifar10'."
            )
    def _create_dataloaders(self):
        """
        Creates PyTorch DataLoaders for training and testing.
        """

        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=SHUFFLE,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )

        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )
    def get_train_loader(self):
        """
        Returns the training DataLoader.
        """
        return self.train_loader


    def get_test_loader(self):
        """
        Returns the testing DataLoader.
        """
        return self.test_loader
    

    def get_dataset_info(self):
        """
        Returns dataset metadata.
        """
        return self.info
if __name__ == "__main__":

    print("=" * 60)
    print(" Image Denoising Dataset Manager")
    print("=" * 60)

    for dataset_name in ["mnist", "cifar10"]:

        print(f"\nLoading {dataset_name.upper()}...")

        manager = DatasetManager(dataset_name)

        info = manager.get_dataset_info()

        print(f"Dataset      : {info.name}")
        print(f"Channels     : {info.channels}")
        print(f"Image Size   : {info.image_size}")
        print(f"Classes      : {info.num_classes}")
        print(f"Train Images : {info.train_size}")
        print(f"Test Images  : {info.test_size}")

        train_loader = manager.get_train_loader()

        images, labels = next(iter(train_loader))

        print(f"Batch Shape  : {images.shape}")
        print(f"Labels Shape : {labels.shape}")

        print("-" * 60)

    print("\nDataset loading completed successfully.")