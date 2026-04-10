#!/usr/bin/env python3
"""Generate all DL course materials: feature maps, gradients, training curves, 
layer specs, receptive field analysis, and slide content."""

import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import create_dataloaders
from src.data.prompt_pool import PROMPT_POOL, tokenize, MAX_PROMPT_LEN
from src.model.promptseg import PromptSegLite
from src.losses.dice_bce import DiceBCELoss
from src.utils.hooks import FeatureGradientCapture, generate_layer_specs
from src.utils.visualization import (
    visualize_feature_maps, visualize_predictions, plot_training_curves
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_receptive_field(model_cfg):
    """Analytically compute receptive field at bottleneck."""
    # Stem: 3x3, stride 2
    # Stages 1-4: each has 3x3 DW conv, stride 2 (+ 1x1 PW which doesn't change RF)
    layers = [
        {"name": "stem", "k": 3, "s": 2, "p": 1},
        {"name": "stage1_dw", "k": 3, "s": 2, "p": 1},
        {"name": "stage2_dw", "k": 3, "s": 2, "p": 1},
        {"name": "stage3_dw", "k": 3, "s": 2, "p": 1},
        {"name": "stage4_dw", "k": 3, "s": 2, "p": 1},
    ]

    rf = 1
    stride_product = 1
    results = []

    for layer in layers:
        k = layer["k"]
        s = layer["s"]
        rf = rf + (k - 1) * stride_product
        stride_product *= s
        results.append({
            "name": layer["name"],
            "kernel": k,
            "stride": s,
            "rf_after": rf,
            "cumulative_stride": stride_product,
        })

    return results, rf


def compute_flops(model, input_size=(1, 3, 256, 256), token_len=None):
    """Estimate FLOPs by counting multiply-add operations."""
    if token_len is None:
        token_len = MAX_PROMPT_LEN

    total_flops = 0
    hook_handles = []

    def conv_flops_hook(module, input, output):
        nonlocal total_flops
        batch = output.shape[0]
        out_h, out_w = output.shape[2], output.shape[3]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        total_flops += batch * module.out_channels * out_h * out_w * kernel_ops

    def linear_flops_hook(module, input, output):
        nonlocal total_flops
        batch = input[0].shape[0] if len(input[0].shape) > 1 else 1
        total_flops += batch * module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hook_handles.append(m.register_forward_hook(conv_flops_hook))
        elif isinstance(m, nn.Linear):
            hook_handles.append(m.register_forward_hook(linear_flops_hook))

    device = next(model.parameters()).device
    dummy_img = torch.randn(*input_size, device=device)
    dummy_tok = torch.ones(input_size[0], token_len, dtype=torch.long, device=device)

    with torch.no_grad():
        model(dummy_img, dummy_tok)

    for h in hook_handles:
        h.remove()

    return total_flops


def generate_parameter_breakdown(model):
    """Break down parameters by component."""
    components = {
        "text_encoder": 0,
        "vision_encoder.stem": 0,
        "vision_encoder.stages": 0,
        "decoder": 0,
        "other": 0,
    }

    for name, param in model.named_parameters():
        matched = False
        for comp_key in ["text_encoder", "decoder"]:
            if comp_key in name:
                components[comp_key] += param.numel()
                matched = True
                break
        if not matched:
            if "vision_encoder" in name and "stem" in name:
                components["vision_encoder.stem"] += param.numel()
            elif "vision_encoder" in name:
                components["vision_encoder.stages"] += param.numel()
            else:
                components["other"] += param.numel()

    return components


def main():
    cfg = yaml.safe_load(open("config/default.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    figures_dir = "report/figures"
    os.makedirs(figures_dir, exist_ok=True)

    # Load best model
    ckpt_path = os.path.join(cfg.get("checkpoint_dir", "checkpoints"), "best_model.pt")
    if not os.path.exists(ckpt_path):
        ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
        best_files = sorted([
            f for f in os.listdir(ckpt_dir) if f.endswith("_best.pt")
        ])
        if best_files:
            ckpt_path = os.path.join(ckpt_dir, best_files[-1])
        else:
            ckpt_path = os.path.join(ckpt_dir, "latest.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint at {ckpt_path}")
        sys.exit(1)

    model = PromptSegLite(cfg.get("model", {})).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded best model from epoch {epoch}")

    # ================================================================
    # Slide 1: Input Tensor
    # ================================================================
    print("\n--- Slide 1: Input Tensor ---")
    slide1 = {
        "image_tensor": "(B, 3, 256, 256) float32",
        "prompt_tokens": f"(B, {MAX_PROMPT_LEN}) int64",
        "output_tensor": "(B, 1, 256, 256) float32 (logits → sigmoid → mask)",
        "normalization": "ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
        "tokenization": "Character-level, vocab_size=14 unique chars + PAD",
    }
    for k, v in slide1.items():
        print(f"  {k}: {v}")

    # ================================================================
    # Slide 2: Major Blocks
    # ================================================================
    print("\n--- Slide 2: Major Blocks ---")
    slide2 = {
        "block1": "Text Encoder: Embedding(14, 64) → BiLSTM(64, 128, bidirectional) → FC(256) → FiLM generators",
        "block2": "Vision Encoder: Stem(3→32,s2) + 4 MobileBlocks (DSConv + SE + FiLM): 32→64→128→256→512",
        "block3": "U-Net Decoder: 3 skip-connected stages (512+256→256, 256+128→128, 128+64→64) + upsample(64→32) + head(32→1)",
        "conditioning": "FiLM (Feature-wise Linear Modulation): γ * x + β at each encoder/decoder stage",
        "attention": "SE (Squeeze-and-Excitation) channel attention in each encoder stage",
    }
    for k, v in slide2.items():
        print(f"  {k}: {v}")

    # ================================================================
    # Slides 3-5: Layer Specs
    # ================================================================
    print("\n--- Slides 3-5: Layer Specifications ---")
    dummy_img = torch.randn(1, 3, 256, 256, device=device)
    dummy_tok = torch.ones(1, MAX_PROMPT_LEN, dtype=torch.long, device=device)
    specs = generate_layer_specs(model, dummy_img, dummy_tok)
    specs_path = os.path.join("report", "layer_specs.json")
    with open(specs_path, "w") as f:
        json.dump(specs, f, indent=2, default=str)
    print(f"  Generated {len(specs)} layer specs → {specs_path}")

    # ================================================================
    # Slide 6: Feature Maps & Gradients
    # ================================================================
    print("\n--- Slide 6: Feature Maps & Gradients ---")
    capture = FeatureGradientCapture()
    layer_names = [
        "vision_encoder.stage1.se",   # Encoder Stage 1 SE output (shallow features)
        "decoder.dec2.conv",           # Decoder Stage 2 conv output (deep features)
    ]
    capture.register(model, layer_names)

    # Get a real sample
    _, val_loader = create_dataloaders(cfg)
    images, masks, token_ids, class_idx = next(iter(val_loader))
    images = images.to(device)
    masks = masks.to(device)
    token_ids = token_ids.to(device)

    model.train()  # Need gradients
    criterion = DiceBCELoss()
    logits = model(images, token_ids)
    loss = criterion(logits, masks)
    loss.backward()

    for ln in layer_names:
        visualize_feature_maps(capture, ln, os.path.join(figures_dir, "feature_maps"))
        print(f"  Saved feature map visualization for {ln}")

    capture.remove_hooks()
    model.eval()

    # ================================================================
    # Slide 7: Loss & Optimizer
    # ================================================================
    print("\n--- Slide 7: Loss & Optimizer ---")
    slide7 = {
        "loss": "DiceBCE = 1.0 * DiceLoss + 1.0 * BCEWithLogits",
        "optimizer": f"Adam (lr={cfg['training']['learning_rate']}, wd={cfg['training']['weight_decay']})",
        "scheduler": f"CosineAnnealing (warmup={cfg['training']['warmup_epochs']} epochs, min_lr={cfg['training']['min_lr']})",
        "grad_clip": cfg["training"]["grad_clip"],
        "amp": "Enabled (float16 compute, float32 master weights)",
    }
    for k, v in slide7.items():
        print(f"  {k}: {v}")

    # ================================================================
    # Slide 8: Training Curves
    # ================================================================
    print("\n--- Slide 8: Training Curves ---")
    history_path = os.path.join(cfg.get("log_dir", "runs"), "training_history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(history, os.path.join(figures_dir, "training_curves.png"))
        print(f"  Saved training curves plot")
    else:
        print(f"  [WARN] No training history found at {history_path}")

    # ================================================================
    # Slide 9: Receptive Field & Complexity
    # ================================================================
    print("\n--- Slide 9: Receptive Field & Complexity ---")

    rf_results, total_rf = compute_receptive_field(cfg.get("model", {}))
    print(f"  Receptive field at bottleneck: {total_rf}×{total_rf} pixels")
    print(f"  Coverage of 256×256 input: {total_rf/256*100:.1f}%")

    for r in rf_results:
        print(f"    {r['name']}: RF={r['rf_after']}, stride={r['cumulative_stride']}")

    # FLOPs
    flops = compute_flops(model)
    print(f"  FLOPs (single image): {flops/1e6:.1f}M")

    # Parameter breakdown
    breakdown = generate_parameter_breakdown(model)
    total_params = sum(breakdown.values())
    print(f"  Total parameters: {total_params:,}")
    for comp, count in breakdown.items():
        print(f"    {comp}: {count:,} ({count/total_params*100:.1f}%)")

    # Model size
    model_size_mb = total_params * 4 / 1e6
    print(f"  Model size (FP32): {model_size_mb:.1f} MB")
    print(f"  Model size (FP16): {model_size_mb/2:.1f} MB")

    # Inference speed
    print("\n  Speed benchmark:")
    dummy = torch.randn(1, 3, 256, 256, device=device)
    dummy_tok = torch.ones(1, MAX_PROMPT_LEN, dtype=torch.long, device=device)
    for _ in range(20):
        with torch.no_grad():
            model(dummy, dummy_tok)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        with torch.no_grad():
            model(dummy, dummy_tok)
    torch.cuda.synchronize()
    latency = (time.perf_counter() - t0) / 200 * 1000
    print(f"  L4 GPU (FP32): {latency:.2f} ms/image ({1000/latency:.0f} FPS)")

    # AMP speed
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        with torch.no_grad(), autocast('cuda'):
            model(dummy, dummy_tok)
    torch.cuda.synchronize()
    latency_amp = (time.perf_counter() - t0) / 200 * 1000
    print(f"  L4 GPU (AMP FP16): {latency_amp:.2f} ms/image ({1000/latency_amp:.0f} FPS)")

    # Save slide 9 data
    slide9 = {
        "receptive_field": total_rf,
        "rf_coverage_pct": round(total_rf / 256 * 100, 1),
        "rf_layers": rf_results,
        "flops_millions": round(flops / 1e6, 1),
        "total_params": total_params,
        "param_breakdown": breakdown,
        "model_size_fp32_mb": round(model_size_mb, 1),
        "model_size_fp16_mb": round(model_size_mb / 2, 1),
        "latency_fp32_ms": round(latency, 2),
        "latency_amp_ms": round(latency_amp, 2),
        "fps_fp32": round(1000 / latency),
        "fps_amp": round(1000 / latency_amp),
    }
    with open(os.path.join("report", "slide9_complexity.json"), "w") as f:
        json.dump(slide9, f, indent=2)
    print(f"\n  Saved complexity analysis to report/slide9_complexity.json")

    print("\n" + "=" * 60)
    print("All DL course materials generated!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  report/layer_specs.json")
    print(f"  report/figures/feature_maps/")
    print(f"  report/figures/training_curves.png")
    print(f"  report/slide9_complexity.json")


if __name__ == "__main__":
    main()
