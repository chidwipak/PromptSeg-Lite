"""MobileBlock-inspired lightweight vision encoder with SE and FiLM conditioning."""

import torch
import torch.nn as nn

from src.model.se import SEBlock
from src.model.film import FiLMLayer


class DSConv(nn.Module):
    """Depthwise Separable Convolution block.
    
    Depthwise conv (groups=in_ch) -> BN -> ReLU6 -> Pointwise conv (1x1) -> BN -> ReLU6
    8-9x fewer FLOPs than standard convolution.
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                            groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class EncoderStage(nn.Module):
    """Encoder stage: n x DSConv + SE + FiLM conditioning."""

    def __init__(self, in_ch, out_ch, n_blocks=2, stride=2, se_reduction=4):
        super().__init__()
        layers = [DSConv(in_ch, out_ch, stride=stride)]
        for _ in range(n_blocks - 1):
            layers.append(DSConv(out_ch, out_ch, stride=1))
        self.convs = nn.Sequential(*layers)
        self.se = SEBlock(out_ch, reduction=se_reduction)
        self.film = FiLMLayer()

    def forward(self, x, gamma, beta):
        x = self.convs(x)
        x = self.se(x)
        x = self.film(x, gamma, beta)
        return x


class VisionEncoder(nn.Module):
    """Lightweight MobileBlock-inspired encoder with 4 stages.
    
    Input: (B, 3, 256, 256)
    Stage 0 (stem):  Conv2d(3, 32, 3, stride=2) -> (B, 32, 128, 128)
    Stage 1: DSConv(32→64, stride=2) + SE + FiLM  -> (B, 64, 64, 64)    skip1
    Stage 2: DSConv(64→128, stride=2) + SE + FiLM -> (B, 128, 32, 32)   skip2
    Stage 3: DSConv(128→256, stride=2) + SE + FiLM -> (B, 256, 16, 16)  skip3
    Stage 4: DSConv(256→512, stride=2) + SE + FiLM -> (B, 512, 8, 8)    bottleneck
    """

    def __init__(self, channels=None, se_reduction=4):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256, 512]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU6(inplace=True),
        )

        # Encoder stages
        self.stage1 = EncoderStage(channels[0], channels[1], n_blocks=2,
                                   stride=2, se_reduction=se_reduction)
        self.stage2 = EncoderStage(channels[1], channels[2], n_blocks=3,
                                   stride=2, se_reduction=se_reduction)
        self.stage3 = EncoderStage(channels[2], channels[3], n_blocks=3,
                                   stride=2, se_reduction=se_reduction)
        self.stage4 = EncoderStage(channels[3], channels[4], n_blocks=2,
                                   stride=2, se_reduction=se_reduction)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, film_params):
        """
        Args:
            x: (B, 3, H, W) input image
            film_params: list of (gamma, beta) from text encoder
                film_params[0] for stage1, [1] for stage2, [2] for stage3, [3] for stage4
        Returns:
            skips: [skip1, skip2, skip3] for decoder
            bottleneck: (B, 512, H/32, W/32)
        """
        x = self.stem(x)  # (B, 32, H/2, W/2)

        s1 = self.stage1(x, *film_params[0])   # (B, 64, H/4, W/4)
        s2 = self.stage2(s1, *film_params[1])   # (B, 128, H/8, W/8)
        s3 = self.stage3(s2, *film_params[2])   # (B, 256, H/16, W/16)
        bottleneck = self.stage4(s3, *film_params[3])  # (B, 512, H/32, W/32)

        return [s1, s2, s3], bottleneck
