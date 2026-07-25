"""
predict.py

Run inference using the trained denoising autoencoder.

"""

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from config import CHECKPOINT_DIR, DEVICE, OUTPUT_DIR
from model import DenoisingAutoencoder
from utils import create_directory

# Configuration

IMAGE_PATH = "sample_input.png"      
MODEL_NAME = "mnist_best_model.pth"

OUTPUT_FOLDER = Path(OUTPUT_DIR) / "predictions"

IMAGE_SIZE = (28, 28)              

# Load Model

def load_model():

    checkpoint_path = Path(CHECKPOINT_DIR) / MODEL_NAME

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    model = DenoisingAutoencoder().to(DEVICE)

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )
    )

    model.eval()

    return model


# Image Transform

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


# Prediction

def predict():

    create_directory(OUTPUT_FOLDER)

    model = load_model()

    image_path = Path(IMAGE_PATH)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = Image.open(image_path).convert("L")

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor)

    input_path = OUTPUT_FOLDER / "input.png"
    output_path = OUTPUT_FOLDER / "denoised.png"

    save_image(image_tensor.cpu(), input_path)
    save_image(output.cpu(), output_path)

    print("=" * 60)
    print("Prediction Completed Successfully!")
    print("=" * 60)
    print(f"Input Image     : {input_path}")
    print(f"Denoised Image  : {output_path}")


# Main

if __name__ == "__main__":
    predict()