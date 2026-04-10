"""PromptSeg-Lite: Full text-conditioned segmentation model."""

import torch
import torch.nn as nn

from src.model.text_encoder import TextEncoder
from src.model.vision_encoder import VisionEncoder
from src.model.decoder import Decoder
from src.model.aspp import ASPP


class PromptSegLite(nn.Module):
    """PromptSeg-Lite: Lightweight text-conditioned binary segmentation model.
    
    Architecture:
        Text Encoder (BiLSTM) -> FiLM parameters
        Vision Encoder (MobileBlock + SE) + FiLM conditioning -> features + skips
        ASPP at bottleneck (optional) -> multi-scale context
        U-Net Decoder + FiLM conditioning + Deep Supervision -> binary mask
    
    Input:
        image: (B, 3, H, W) float32
        token_ids: (B, seq_len) int64
    Output:
        logits: (B, 1, H, W) float32 (pre-sigmoid) or dict with deep supervision
    """

    def __init__(self, cfg=None):
        super().__init__()

        # Default config
        if cfg is None:
            cfg = {}
        
        text_cfg = cfg.get("text_encoder", {})
        vis_cfg = cfg.get("vision_encoder", {})
        dec_cfg = cfg.get("decoder", {})

        enc_channels = vis_cfg.get("channels", [32, 64, 128, 256, 512])
        dec_channels = dec_cfg.get("channels", [256, 128, 64, 32])
        se_reduction = vis_cfg.get("se_reduction", 4)
        use_aspp = cfg.get("use_aspp", False)
        deep_supervision = cfg.get("deep_supervision", False)

        # FiLM stages: 4 encoder + 3 decoder = 7 total
        film_stage_channels = (
            enc_channels[1:] +    # [64, 128, 256, 512] for encoder
            dec_channels[:3]       # [256, 128, 64] for decoder
        )

        self.text_encoder = TextEncoder(
            vocab_size=text_cfg.get("vocab_size", 128),
            embed_dim=text_cfg.get("embed_dim", 64),
            hidden_size=text_cfg.get("hidden_size", 128),
            num_layers=text_cfg.get("num_layers", 1),
            bidirectional=text_cfg.get("bidirectional", True),
            stage_channels=film_stage_channels,
        )

        self.vision_encoder = VisionEncoder(
            channels=enc_channels,
            se_reduction=se_reduction,
        )

        # Optional ASPP at bottleneck
        self.use_aspp = use_aspp
        if use_aspp:
            aspp_mid = cfg.get("aspp_mid_ch", 128)
            self.aspp = ASPP(
                in_ch=enc_channels[-1],
                out_ch=enc_channels[-1],
                rates=(6, 12, 18),
                mid_ch=aspp_mid,
            )

        self.decoder = Decoder(
            encoder_channels=enc_channels[1:],  # [64, 128, 256, 512]
            decoder_channels=dec_channels,
            deep_supervision=deep_supervision,
        )

    def forward(self, image, token_ids):
        """
        Args:
            image: (B, 3, H, W) normalized input image
            token_ids: (B, seq_len) character-level token IDs
        Returns:
            logits: (B, 1, H, W) pre-sigmoid logits, or dict with deep supervision
        """
        # Text encoding -> FiLM parameters
        film_params = self.text_encoder(token_ids)

        # Vision encoding
        skips, bottleneck = self.vision_encoder(image, film_params[:4])

        # ASPP at bottleneck for multi-scale context
        if self.use_aspp:
            bottleneck = self.aspp(bottleneck)

        # Decoding (with optional deep supervision)
        output = self.decoder(skips, bottleneck, film_params[4:7])

        return output

    def predict(self, image, token_ids, threshold=0.5):
        """Run inference and return binary mask."""
        self.eval()
        with torch.no_grad():
            output = self.forward(image, token_ids)
            # Handle deep supervision dict
            logits = output['main'] if isinstance(output, dict) else output
            probs = torch.sigmoid(logits)
            mask = (probs > threshold).float()
        return mask, probs

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
