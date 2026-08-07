# Multi-Backbone Feature Fusion for Image Classification

## Overview

This project implements a robust hybrid image processing framework integrating:

- Classical Image Processing
- Deep Learning
- Adaptive Feature Fusion

The architecture is designed to improve robustness under degraded image conditions such as:

- Gaussian Noise
- Salt & Pepper Noise
- Motion Blur

The system combines handcrafted structural information from Sobel edge extraction with semantic representations learned through CNNs.

---

# Motivation

Traditional CNNs perform well on clean datasets but often fail under noisy and degraded conditions.

Classical image processing methods are comparatively more robust to image corruption but lack adaptive learning capability.

This project combines both approaches using:

- Dual-branch feature extraction
- Sobel edge processing
- Adaptive fusion
- Noise-aware learning

---

# Architecture

```text
Input Image
      ↓
Noise Injection
      ↓
 ┌──────────────────────┐
 │                      │
 ↓                      ↓
Raw Noisy Image    Sobel Edge Image
 ↓                      ↓
CNN Branch         CNN Branch
 ↓                      ↓
Deep Features     Structural Features
        ↓
   Adaptive Fusion
        ↓
 Fully Connected Layer
        ↓
    Classification
```

---

# Mathematical Formulation

## Classical Processing

\[
X_c = \Phi(X)
\]

Where:
- \(X\) = input image
- \(\Phi\) = classical image processing function

---

## Deep Feature Extraction

\[
F_d = D(X)
\]

\[
F_c = D(X_c)
\]

Where:
- \(D\) = CNN feature extractor

---

## Adaptive Fusion

\[
F = \alpha F_d + (1-\alpha)F_c
\]

Where:
- \(F_d\) = deep features
- \(F_c\) = classical features
- \(\alpha\) = dynamically learned fusion weight

---

# Dataset

## CIFAR-10

- 60,000 RGB Images
- 10 Classes
- 32×32 Resolution

Classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

# Project Structure

```text
Hybrid-Image-Framework/
│
├── data/
│
├── src/
│   ├── classical/
│   │   └── filters.py
│   │
│   ├── fusion/
│   │   └── weighted_fusion.py
│   │
│   ├── models/
│   │   ├── hybrid_cnn.py
│   │   └── baseline_cnn.py
│   │
│   ├── training/
│   │   └── trainer.py
│   │
│   └── utils/
│       └── dataloader.py
│
├── train.py
├── train_baseline.py
├── evaluate.py
├── visualize_predictions.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

## Clone Repository

```bash
git clone <repo_url>
cd Hybrid-Image-Framework
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Training

## Train Hybrid Model

```bash
python train.py
```

---

## Train Baseline CNN

```bash
python train_baseline.py
```

---

# Evaluation

```bash
python evaluate.py
```

This compares:
- Baseline CNN
- Hybrid CNN

under noisy conditions.

---

# Visualization

```bash
python visualize_predictions.py
```

Displays:
- noisy images
- predictions
- ground truth labels

---

# Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- CIFAR-10

---

# Key Features

- Hybrid Classical + Deep Learning Architecture
- Sobel Edge Feature Extraction
- Adaptive Feature Fusion
- Multiple Noise Augmentation Techniques
- Robustness-Oriented Training
- Visualization Pipeline
- Baseline CNN Comparison

---

# Current Research Direction

This project explores:

- Robust Vision Systems
- Classical + Deep Learning Integration
- Adaptive Feature Fusion
- Noise-Aware Representation Learning

---

# Future Improvements

Potential future enhancements:

- ResNet Backbone
- Vision Transformers
- Attention-Based Fusion
- Adversarial Robustness
- Medical Image Applications
- Explainable AI Integration

---

# Results

The hybrid framework demonstrates:

- successful dual-branch learning
- adaptive feature fusion
- robustness-oriented architecture
- structural feature integration

The project serves as a strong foundation for further research in robust computer vision systems.

---

# Author

Prathamesh Ranaware
