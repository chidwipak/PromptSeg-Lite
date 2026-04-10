"""Feature-wise Linear Modulation (FiLM) conditioning layer."""

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: gamma * x + beta.
    
    Modulates feature maps channel-wise using conditioning from text encoder.
    Zero computational overhead beyond the affine transform.
    """

    def forward(self, x, gamma, beta):
        """
        Args:
            x: (B, C, H, W) feature map
            gamma: (B, C) scale factors from text encoder
            beta: (B, C) shift factors from text encoder
        Returns:
            (B, C, H, W) modulated feature map
        """
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)
        return gamma * x + beta


class FiLMGenerator(nn.Module):
    """Generates FiLM gamma/beta parameters from a conditioning vector."""

    def __init__(self, cond_dim, num_channels):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, num_channels)
        self.beta_fc = nn.Linear(cond_dim, num_channels)

        # Initialize gamma close to 1, beta close to 0
        nn.init.ones_(self.gamma_fc.bias)
        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, cond):
        """
        Args:
            cond: (B, cond_dim) conditioning vector
        Returns:
            gamma: (B, num_channels)
            beta: (B, num_channels)
        """
        gamma = self.gamma_fc(cond)
        beta = self.beta_fc(cond)
        return gamma, beta
