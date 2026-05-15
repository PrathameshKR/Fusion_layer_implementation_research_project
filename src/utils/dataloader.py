from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import cv2
import numpy as np

# ----------------------------
# Add Gaussian Noise
# ----------------------------

def add_gaussian_noise(image, mean=0, std=25):

    noise = np.random.normal(
        mean,
        std,
        image.shape
    ).astype(np.float32)

    noisy = image.astype(np.float32) + noise

    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)

# ----------------------------
# Hybrid Transform
# ----------------------------

class HybridTransform:

    def __call__(self, image):

        image_np = np.array(image)

        # --------------------------------
        # RAW NOISY IMAGE
        # --------------------------------

        noisy_image = add_gaussian_noise(
            image_np
        )

        raw_tensor = transforms.ToTensor()(
            noisy_image
        )

        # --------------------------------
        # FILTERED IMAGE
        # --------------------------------

        filtered = cv2.GaussianBlur(
            noisy_image,
            (3,3),
            0
        )

        filtered_tensor = transforms.ToTensor()(
            filtered
        )

        return raw_tensor, filtered_tensor

# ----------------------------
# Dataset Wrapper
# ----------------------------

class HybridDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):

        self.dataset = dataset

        self.transform = HybridTransform()

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, idx):

        image, label = self.dataset[idx]

        raw_img, filtered_img = self.transform(image)

        return raw_img, filtered_img, label

# ----------------------------
# Dataloader
# ----------------------------

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

    train_dataset = HybridDataset(base_train)

    test_dataset = HybridDataset(base_test)

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