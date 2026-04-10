"""Phase 2 Loss Functions: Focal Tversky + Boundary Loss + OHEM.

Designed to aggressively improve crack segmentation (IoU 0.60 -> 0.70+)
by penalizing false negatives more heavily and focusing on hard examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss for handling extreme class imbalance.

    Tversky index generalizes Dice by weighting FP and FN differently:
        TI = TP / (TP + alpha*FN + beta*FP)

    alpha > beta penalizes false negatives more (missed crack pixels).
    Focal exponent gamma < 1.0 down-weights easy examples.

    Reference: Abraham & Khan, 2019 - "A Novel Focal Tversky Loss Function
    with Improved Attention U-Net for Lesion Segmentation"
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1.0):
        super().__init__()
        self.alpha = alpha   # FN weight (higher = penalize missed detections more)
        self.beta = beta     # FP weight
        self.gamma = gamma   # Focal exponent (<1 focuses on hard examples)
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) raw logits (pre-sigmoid)
            targets: (B, 1, H, W) binary ground truth {0, 1}
        Returns:
            Scalar loss
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        tp = (probs_flat * targets_flat).sum(dim=1)
        fn = ((1 - probs_flat) * targets_flat).sum(dim=1)
        fp = (probs_flat * (1 - targets_flat)).sum(dim=1)

        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )

        # Focal modulation: (1 - TI)^gamma
        focal_tversky = (1 - tversky_index).pow(self.gamma)
        return focal_tversky.mean()


class BoundaryLoss(nn.Module):
    """Boundary-aware loss that emphasizes edge regions.

    Computes a boundary distance map from targets using morphological operations,
    then applies weighted BCE focusing on boundary pixels.
    """

    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        # Create dilation kernel (fixed, not trainable)
        kernel = torch.ones(1, 1, kernel_size, kernel_size)
        self.register_buffer('kernel', kernel)

    def _get_boundary(self, mask):
        """Extract boundary pixels via morphological dilation - erosion."""
        # mask: (B, 1, H, W) float {0, 1}
        pad = self.kernel_size // 2
        kernel = self.kernel.to(mask.device, mask.dtype)

        dilated = F.conv2d(mask, kernel, padding=pad)
        dilated = (dilated > 0).float()

        eroded = F.conv2d(mask, kernel, padding=pad)
        eroded = (eroded >= self.kernel_size * self.kernel_size).float()

        boundary = dilated - eroded
        return boundary

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) raw logits
            targets: (B, 1, H, W) binary ground truth
        Returns:
            Boundary-weighted BCE loss
        """
        boundary = self._get_boundary(targets)
        # Weight: 1.0 for non-boundary, extra weight for boundary
        weight = 1.0 + boundary * 4.0  # 5x weight on boundary pixels

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weighted_bce = (bce * weight).mean()
        return weighted_bce


class OHEMLoss(nn.Module):
    """Online Hard Example Mining loss.

    Keeps only the top-k% hardest pixel losses, forcing the model to
    focus on the most difficult regions (typically thin cracks, edges,
    ambiguous textures).
    """

    def __init__(self, top_k_ratio=0.25):
        super().__init__()
        self.top_k_ratio = top_k_ratio

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W)
            targets: (B, 1, H, W)
        Returns:
            Mean of top_k hardest pixel losses
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        bce_flat = bce.view(-1)

        k = max(1, int(bce_flat.numel() * self.top_k_ratio))
        topk_losses, _ = torch.topk(bce_flat, k)
        return topk_losses.mean()


class CompoundV2Loss(nn.Module):
    """Phase 2 Compound Loss: Focal Tversky + Boundary + OHEM.

    L = w_ft * FocalTversky + w_boundary * BoundaryLoss + w_ohem * OHEMLoss

    Default weights tuned for crack segmentation improvement:
    - Focal Tversky (alpha=0.7): penalize missed crack pixels
    - Boundary Loss: improve edge delineation
    - OHEM (top 25%): focus on hard pixels
    """

    def __init__(self, ft_weight=1.0, boundary_weight=0.5, ohem_weight=0.5,
                 ft_alpha=0.7, ft_beta=0.3, ft_gamma=0.75,
                 ohem_ratio=0.25):
        super().__init__()
        self.ft_weight = ft_weight
        self.boundary_weight = boundary_weight
        self.ohem_weight = ohem_weight

        self.focal_tversky = FocalTverskyLoss(
            alpha=ft_alpha, beta=ft_beta, gamma=ft_gamma
        )
        self.boundary_loss = BoundaryLoss(kernel_size=3)
        self.ohem_loss = OHEMLoss(top_k_ratio=ohem_ratio)

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) or list of logits for deep supervision
            targets: (B, 1, H, W) binary ground truth
        Returns:
            Scalar compound loss
        """
        ft = self.focal_tversky(logits, targets)
        boundary = self.boundary_loss(logits, targets)
        ohem = self.ohem_loss(logits, targets)

        return (self.ft_weight * ft +
                self.boundary_weight * boundary +
                self.ohem_weight * ohem)


class DeepSupervisionLoss(nn.Module):
    """Wrapper that applies loss at multiple scales for deep supervision.

    Main output gets full weight, auxiliary outputs get reduced weight.
    Aux targets are downsampled to match aux output resolution.
    """

    def __init__(self, base_loss, aux_weights=(0.4, 0.2)):
        super().__init__()
        self.base_loss = base_loss
        self.aux_weights = aux_weights

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'main' and optionally 'aux1', 'aux2' logits
            targets: (B, 1, H, W) full-resolution ground truth
        Returns:
            Weighted sum of losses at all scales
        """
        if isinstance(outputs, torch.Tensor):
            # No deep supervision, just compute main loss
            return self.base_loss(outputs, targets)

        # Main loss at full resolution
        main_logits = outputs['main']
        total_loss = self.base_loss(main_logits, targets)

        # Auxiliary losses at lower resolutions
        aux_keys = ['aux1', 'aux2']
        for i, key in enumerate(aux_keys):
            if key in outputs and i < len(self.aux_weights):
                aux_logits = outputs[key]
                # Downsample target to match aux resolution
                aux_h, aux_w = aux_logits.shape[2:]
                aux_targets = F.interpolate(
                    targets, size=(aux_h, aux_w),
                    mode='nearest'
                )
                aux_loss = self.base_loss(aux_logits, aux_targets)
                total_loss = total_loss + self.aux_weights[i] * aux_loss

        return total_loss
