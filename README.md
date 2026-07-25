# Image Denoising using Convolutional Autoencoder (PyTorch)

A deep learning project that removes synthetic noise from images using a **Convolutional Autoencoder** implemented with **PyTorch**. The model is trained to reconstruct clean images from noisy inputs and is evaluated using standard image quality metrics.

---

## Features

- Convolutional Autoencoder architecture
- Supports **MNIST** and **CIFAR-10**
- Three noise types
  - Gaussian Noise
  - Salt & Pepper Noise
  - Speckle Noise
- GPU acceleration (CUDA / Apple Silicon MPS)
- Automatic checkpoint saving
- Evaluation using:
  - Mean Squared Error (MSE)
  - Peak Signal-to-Noise Ratio (PSNR)
  - Structural Similarity Index (SSIM)
- Prediction on custom images
- Jupyter notebook for visualization

---

## Project Structure

```text
ImageDenoising/
│
├── notebooks/
│   └── dataset_visualization.ipynb
│
├── src/
│   ├── checkpoints/
│   ├── data/
│   ├── outputs/
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── generate_noisy_images.py
│   ├── metrics.py
│   ├── model.py
│   ├── noise.py
│   ├── predict.py
│   ├── sample_input.png
│   ├── train.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Model Architecture

The denoising model is a Convolutional Autoencoder consisting of:

- Convolution Layers
- Batch Normalization
- Max Pooling
- Transposed Convolution Layers
- Sigmoid Output Layer

The encoder learns a compact latent representation of the noisy image, while the decoder reconstructs a clean version.

---

## Workflow

```text
Clean Image
      │
      ▼
Add Artificial Noise
      │
      ▼
Convolutional Autoencoder
      │
      ▼
Denoised Image
      │
      ▼
Evaluation (MSE • PSNR • SSIM)
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ImageDenoising.git

cd ImageDenoising
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Training

```bash
cd src

python train.py
```

The best model checkpoint will be saved inside:

```text
src/checkpoints/
```

---

## Evaluation

```bash
python evaluate.py
```

Example evaluation metrics:

```text
Dataset        : MNIST
Samples Tested : 10000

Average MSE    : 0.005440
Average PSNR   : 22.92 dB
Average SSIM   : 0.8979
```

---

## Generate Sample Noisy Images

```bash
python generate_noisy_images.py
```

Generated images are stored in:

```text
src/outputs/noisy_images/
```

---

## Prediction on Custom Images

Place an input image in the `src` directory (or update the image path in `predict.py`) and run:

```bash
python predict.py
```

The prediction results are saved in:

```text
src/outputs/predictions/
```

---

## Dataset Visualization

The notebook located in

```text
notebooks/dataset_visualization.ipynb
```

demonstrates:

- Original images
- Different noise types
- Model predictions
- Image reconstruction results
- Evaluation summary

---

## Technologies Used

- Python
- PyTorch
- TorchVision
- NumPy
- Matplotlib
- Pillow
- scikit-image

---

## Future Work

- Extend the model to support real-world noisy image datasets.
- Explore advanced denoising architectures such as U-Net and DnCNN.
- Train on higher-resolution images.
- Develop an interactive web interface for image denoising.

---

## Author

**Chetana Ingle**

Computer Engineering Graduate | AI & Machine Learning Enthusiast

---
