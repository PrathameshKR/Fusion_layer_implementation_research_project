import torch
import torch.nn as nn

class WeightedFusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, deep_features, classical_features):

        fused = (
            self.alpha * deep_features +
            (1 - self.alpha) * classical_features
        )

        return fused