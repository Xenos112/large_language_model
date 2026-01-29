"""
Layers module for the model architecture
    - Implementation of RMSNorm
"""

import torch  # FIX: torch is installed yet it still throws error
from torch.nn import nn

from src.config.config import ModelConfig


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
        - Implementation of RMSNorm
        - Initializes the weight parameter with ones
        - Implements the forward pass for RMSNorm
    """

    def __init__(self, hidden_dim=ModelConfig.hidden_dim, epsilon=ModelConfig.epsilon):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.epsilon = epsilon

    def forward(self, x):
        return (
            self.weight
            * x
            / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.epsilon)
        )


# TODO: consider using DeepNorm from google for more stability
