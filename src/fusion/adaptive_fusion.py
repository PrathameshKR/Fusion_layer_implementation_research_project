import torch
import torch.nn as nn

class AdaptiveFusion(nn.Module):

    def __init__(self, channels=64):

        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Linear(
                channels * 2,
                channels
            ),

            nn.ReLU(),

            nn.Linear(
                channels,
                1
            ),

            nn.Sigmoid()
        )

    def forward(
        self,
        deep_features,
        classical_features
    ):

        deep_vec = self.global_pool(
            deep_features
        )

        classical_vec = self.global_pool(
            classical_features
        )

        deep_vec = deep_vec.view(
            deep_vec.size(0),
            -1
        )

        classical_vec = classical_vec.view(
            classical_vec.size(0),
            -1
        )

        combined = torch.cat(
            [deep_vec, classical_vec],
            dim=1
        )

        alpha = self.fc(combined)

        alpha = alpha.view(
            -1,
            1,
            1,
            1
        )

        fused = (
            alpha * deep_features +
            (1 - alpha) * classical_features
        )

        return fused