"""U-Net decoder with FiLM-conditioned skip connections and deep supervision."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.vision_encoder import DSConv
from src.model.film import FiLMLayer


class DecoderStage(nn.Module):
    """Decoder stage: Upsample + Concat skip + DSConv + FiLM."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = DSConv(in_ch + skip_ch, out_ch, stride=1)
        self.film = FiLMLayer()

    def forward(self, x, skip, gamma, beta):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.film(x, gamma, beta)
        return x


class Decoder(nn.Module):
    """U-Net decoder with 4 upsample stages and optional deep supervision.
    
    Bottleneck (512, 8, 8)
    -> Stage4: Up + cat(skip3=256) -> DSConv(768->256) + FiLM -> (256, 16, 16)  [aux1]
    -> Stage3: Up + cat(skip2=128) -> DSConv(384->128) + FiLM -> (128, 32, 32)  [aux2]
    -> Stage2: Up + cat(skip1=64)  -> DSConv(192->64)  + FiLM -> (64, 64, 64)
    -> Stage1: Up + DSConv(64->32) -> (32, 128, 128)
    -> Head:   Up + Conv(32->1) -> (1, 256, 256)  [main]
    """

    def __init__(self, encoder_channels=None, decoder_channels=None,
                 deep_supervision=False):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512]
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]

        self.deep_supervision = deep_supervision

        # Decoder stages (with skip connections)
        self.dec4 = DecoderStage(encoder_channels[3], encoder_channels[2],
                                 decoder_channels[0])
        self.dec3 = DecoderStage(decoder_channels[0], encoder_channels[1],
                                 decoder_channels[1])
        self.dec2 = DecoderStage(decoder_channels[1], encoder_channels[0],
                                 decoder_channels[2])

        # Final upsample stages (no skip connection)
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DSConv(decoder_channels[2], decoder_channels[3], stride=1),
        )

        # Output head
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels[3], 1, kernel_size=1),
        )

        # Deep supervision auxiliary heads
        if deep_supervision:
            self.aux_head1 = nn.Conv2d(decoder_channels[0], 1, kernel_size=1)  # dec4 output
            self.aux_head2 = nn.Conv2d(decoder_channels[1], 1, kernel_size=1)  # dec3 output

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, skips, bottleneck, film_params):
        """
        Args:
            skips: [skip1, skip2, skip3] from encoder
            bottleneck: (B, 512, H/32, W/32)
            film_params: list of (gamma, beta) for decoder stages
                film_params[0] for dec4, [1] for dec3, [2] for dec2
        Returns:
            If deep_supervision=True (training):
                dict {'main': (B,1,H,W), 'aux1': (B,1,H4,W4), 'aux2': (B,1,H8,W8)}
            Else:
                (B, 1, H, W) logits (pre-sigmoid)
        """
        s1, s2, s3 = skips

        x4 = self.dec4(bottleneck, s3, *film_params[0])  # (B, 256, H/16, W/16)
        x3 = self.dec3(x4, s2, *film_params[1])           # (B, 128, H/8, W/8)
        x2 = self.dec2(x3, s1, *film_params[2])           # (B, 64, H/4, W/4)
        x1 = self.dec1(x2)                                 # (B, 32, H/2, W/2)
        main = self.head(x1)                                # (B, 1, H, W)

        if self.deep_supervision and self.training:
            aux1 = self.aux_head1(x4)  # (B, 1, H/16, W/16)
            aux2 = self.aux_head2(x3)  # (B, 1, H/8, W/8)
            return {'main': main, 'aux1': aux1, 'aux2': aux2}

        return main
