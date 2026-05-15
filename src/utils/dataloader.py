from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import cv2
import numpy as np
import random

from src.classical.filters import sobel_edges

# -----------------------------------
# Gaussian Noise
# -----------------------------------

def add_gaussian_noise(image, mean=0, std=50):

    noise = np.random.normal(
        mean,
        std,
        image.shape
    ).astype(np.float32)

    noisy = image.astype(np.float32) + noise

    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)

# -----------------------------------
# Salt & Pepper Noise
# -----------------------------------

def add_salt_pepper_noise(image, amount=0.04):

    noisy = image.copy()

    num_salt = np.ceil(
        amount * image.size * 0.5
    )

    coords = [
        np.random.randint(
            0,
            i - 1,
            int(num_salt)
        )
        for i in image.shape
    ]

    noisy[tuple(coords)] = 255

    num_pepper = np.ceil(
        amount * image.size * 0.5
    )

    coords = [
        np.random.randint(
            0,
            i - 1,
            int(num_pepper)
        )
        for i in image.shape
    ]

    noisy[tuple(coords)] = 0

    return noisy

# -----------------------------------
# Motion Blur
# -----------------------------------

def add_motion_blur(image, kernel_size=5):

    kernel = np.zeros(
        (kernel_size, kernel_size)
    )

    kernel[
        int((kernel_size - 1)/2), :
    ] = np.ones(kernel_size)

    kernel = kernel / kernel_size

    blurred = cv2.filter2D(
        image,
        -1,
        kernel
    )

    return blurred

# -----------------------------------
# Hybrid Transform
# -----------------------------------

class HybridTransform:

    def __call__(self, image):

        image_np = np.array(image)

        # -----------------------------
        # RANDOM NOISE TYPE
        # -----------------------------

        noise_type = random.choice([
            "gaussian",
            "salt_pepper",
            "motion"
        ])

        if noise_type == "gaussian":

            noisy_image = add_gaussian_noise(
                image_np
            )

        elif noise_type == "salt_pepper":

            noisy_image = add_salt_pepper_noise(
                image_np
            )

        else:

            noisy_image = add_motion_blur(
                image_np
            )

        # -----------------------------
        # RAW NOISY BRANCH
        # -----------------------------

        raw_tensor = transforms.ToTensor()(
            noisy_image
        )

        # -----------------------------
        # SOBEL EDGE BRANCH
        # -----------------------------

        filtered = sobel_edges(noisy_image)

        filtered_tensor = transforms.ToTensor()(
            filtered
        )

        return raw_tensor, filtered_tensor

# -----------------------------------
# Dataset Wrapper
# -----------------------------------

class HybridDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):

        self.dataset = dataset

        self.transform = HybridTransform()

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, idx):

        image, label = self.dataset[idx]

        raw_img, filtered_img = self.transform(
            image
        )

        return raw_img, filtered_img, label

# -----------------------------------
# Dataloader
# -----------------------------------

def get_dataloaders(batch_size=64):

    base_train = datasets.CIFAR10(
        root="data",
        train=True,
        download=True
    )

    base_test = datasets.CIFAR10(
        root="data",
        train=False,
        download=True
    )

    train_dataset = HybridDataset(
        base_train
    )

    test_dataset = HybridDataset(
        base_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader