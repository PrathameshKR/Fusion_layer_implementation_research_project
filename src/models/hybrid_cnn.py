import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fusion.adaptive_fusion import AdaptiveFusion

class HybridCNN(nn.Module):

    def __init__(self):

        super(HybridCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2,2)

        self.fusion = AdaptiveFusion(channels=64)

        self.fc1 = nn.Linear(64 * 8 * 8, 256)

        self.fc2 = nn.Linear(256, 10)

    def feature_extractor(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        return x

    def forward(self, raw_img, classical_img):

        deep_features = self.feature_extractor(raw_img)

        classical_features = self.feature_extractor(classical_img)

        fused = self.fusion(
            deep_features,
            classical_features
        )

        fused = fused.view(fused.size(0), -1)

        fused = F.relu(self.fc1(fused))

        output = self.fc2(fused)

        return output