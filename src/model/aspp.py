"""Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context at the bottleneck.

Captures features at multiple receptive field scales using dilated convolutions.
Lightweight variant using depthwise separable convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Module):
    """Depthwise separable atrous convolution."""

    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=dilation, dilation=dilation,
                            groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class ASPP(nn.Module):
    """Lightweight Atrous Spatial Pyramid Pooling.

    Branches:
        1. 1x1 conv (local context)
        2. 3x3 atrous conv, rate=6  (medium context)
        3. 3x3 atrous conv, rate=12 (large context)
        4. 3x3 atrous conv, rate=18 (very large context)
        5. Global average pooling (image-level context)

    All branches are projected to `mid_ch` channels, concatenated,
    then fused via 1x1 conv back to `out_ch`.
    """

    def __init__(self, in_ch, out_ch, rates=(6, 12, 18), mid_ch=128):
        super().__init__()

        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
        )

        # Atrous branches
        self.atrous_branches = nn.ModuleList([
            ASPPConv(in_ch, mid_ch, rate) for rate in rates
        ])

        # Image-level pooling branch
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
        )

        # Fusion: concatenate all branches, then 1x1 conv
        n_branches = 1 + len(rates) + 1  # 1x1 + atrous + pool
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch * n_branches, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(0.1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h, w = x.shape[2:]
        branches = [self.branch1(x)]
        for atrous in self.atrous_branches:
            branches.append(atrous(x))
        pool = self.image_pool(x)
        pool = F.interpolate(pool, size=(h, w), mode='bilinear', align_corners=False)
        branches.append(pool)
        return self.fuse(torch.cat(branches, dim=1))
