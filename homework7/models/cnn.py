#assignment7: diffusion
#date: 11-26-26
#name: Joshua wilson
#description: our cnn

# Used Gemini to help implement the SimpleCNN architecture using nn.Sequential.
from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """A small CNN denoiser for a beginner-friendly diffusion homework.

    This model intentionally avoids U-Net blocks and timestep embeddings.
    It only looks at the noisy image and predicts the added noise directly.
    """

    def __init__(self, hidden_channels: int = 32) -> None:
        super().__init__()
        # ==================== Part 3: Define the CNN Model ====================
        # Students should study or modify the CNN architecture in this section.
        # The model takes a noisy MNIST image as input and predicts the added noise.
        # ======================================================================
        #modified the inputsize 
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)