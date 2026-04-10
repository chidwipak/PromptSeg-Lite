"""Dice + BCE compound loss for binary segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation.
    
    L_dice = 1 - (2 * sum(p*g) + eps) / (sum(p) + sum(g) + eps)
    
    Handles extreme foreground/background imbalance by optimizing
    set-level overlap rather than pixel-level classification.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) raw logits
            targets: (B, 1, H, W) binary ground truth {0, 1}
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """Compound loss: Dice Loss + Binary Cross-Entropy.
    
    L = lambda_dice * L_dice + lambda_bce * L_bce
    
    - BCE provides stable pixel-level gradients even early in training
    - Dice ensures model optimizes for actual evaluation metric (overlap)
    - Together, robust to extreme fg/bg imbalance (cracks < 5% of pixels)
    """

    def __init__(self, dice_weight=1.0, bce_weight=1.0, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) raw logits (pre-sigmoid)
            targets: (B, 1, H, W) binary ground truth {0, 1}
        """
        dice = self.dice_loss(logits, targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        return self.dice_weight * dice + self.bce_weight * bce
