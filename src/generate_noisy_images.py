"""
generate_noisy_images.py

Generate noisy versions of dataset images and save them to disk.

"""

from pathlib import Path

import torchvision.utils as vutils

from config import DEFAULT_DATASET, OUTPUT_DIR
from dataset import DatasetManager
from noise import NoiseGenerator
from utils import create_directory

# Configuration

DATASET_NAME = DEFAULT_DATASET

NOISE_TYPE = "gaussian"
# Options:
# "gaussian"
# "salt_pepper"
# "speckle"

NUM_IMAGES = 20

SAVE_COMPARISON = True

# Main

def main():

    dataset = DatasetManager(DATASET_NAME)

    test_loader = dataset.get_test_loader()

    noise_generator = NoiseGenerator()

    output_dir = Path(OUTPUT_DIR) / "noisy_images"

    original_dir = output_dir / "original"
    noisy_dir = output_dir / NOISE_TYPE

    create_directory(original_dir)
    create_directory(noisy_dir)

    if SAVE_COMPARISON:
        comparison_dir = output_dir / "comparison"
        create_directory(comparison_dir)

    saved = 0

    print("=" * 60)
    print("Generating Noisy Images")
    print("=" * 60)
    print(f"Dataset     : {DATASET_NAME}")
    print(f"Noise Type  : {NOISE_TYPE}")
    print(f"Images      : {NUM_IMAGES}")
    print()

    for images, _ in test_loader:

        noisy_images = noise_generator.add_noise(
            images,
            noise_type=NOISE_TYPE,
        )

        batch_size = images.size(0)

        for i in range(batch_size):

            if saved >= NUM_IMAGES:
                break

            original_path = original_dir / f"image_{saved + 1}.png"
            noisy_path = noisy_dir / f"image_{saved + 1}.png"

            vutils.save_image(images[i], original_path)
            vutils.save_image(noisy_images[i], noisy_path)

            if SAVE_COMPARISON:

                comparison = [images[i], noisy_images[i]]

                comparison_path = (
                    comparison_dir / f"comparison_{saved + 1}.png"
                )

                vutils.save_image(
                    comparison,
                    comparison_path,
                    nrow=2,
                    padding=5,
                )

            saved += 1

            print(f"Saved image {saved}/{NUM_IMAGES}")

        if saved >= NUM_IMAGES:
            break

    print()
    print("=" * 60)
    print("Finished Successfully!")
    print("=" * 60)
    print(f"Images saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()