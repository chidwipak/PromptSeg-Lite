#!/usr/bin/env python3
"""Evaluate PromptSeg-Lite and generate prediction masks."""

import os
import sys
import json
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from PIL import Image
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import PromptSegDataset, create_dataloaders
from src.data.prompt_pool import PROMPT_POOL, tokenize, MAX_PROMPT_LEN
from src.data.transforms import get_val_transforms
from src.model.promptseg import PromptSegLite
from src.metrics.segmentation import MetricTracker, compute_iou, compute_dice


def load_model(checkpoint_path, cfg, device):
    model = PromptSegLite(cfg.get("model", {})).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt.get("epoch", "?")
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(ckpt)
        print("Loaded raw state dict")
    model.eval()
    return model


def evaluate_dataset(model, dataloader, device, use_amp=True):
    """Run evaluation on a dataloader, return per-class and aggregate metrics."""
    tracker = MetricTracker()
    total_loss_not_tracked = 0
    n_batches = 0
    latencies = []

    for images, masks, token_ids, class_idx in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        token_ids = token_ids.to(device, non_blocking=True)
        class_idx = class_idx.to(device, non_blocking=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with autocast('cuda', enabled=use_amp):
            logits = model(images, token_ids)

        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) / images.size(0))

        tracker.update(logits, masks, class_idx)
        n_batches += 1

    metrics = tracker.compute()

    # Inference speed
    avg_latency_ms = np.mean(latencies) * 1000
    metrics["avg_latency_ms"] = avg_latency_ms
    metrics["throughput_fps"] = 1000 / avg_latency_ms if avg_latency_ms > 0 else 0

    return metrics


def generate_prediction_masks(model, cfg, device, output_dir, split="valid",
                              use_amp=True):
    """Generate {image_id}__segment_{class}.png masks for all images."""
    os.makedirs(output_dir, exist_ok=True)

    data_root = cfg["data"]["data_root"]
    image_size = cfg["data"]["image_size"]
    transform = get_val_transforms(image_size)

    datasets_config = [
        {"dir": "drywall_taping", "prompt_class": "taping"},
        {"dir": "cracks", "prompt_class": "crack"},
    ]

    prompt_class_to_label = {
        "taping": "taping_area",
        "crack": "crack",
    }

    count = 0
    for ds_cfg in datasets_config:
        ds_dir = os.path.join(data_root, ds_cfg["dir"], split)
        if not os.path.exists(ds_dir):
            print(f"[WARN] {ds_dir} not found, skipping")
            continue

        prompt_class = ds_cfg["prompt_class"]
        prompt_text = PROMPT_POOL[prompt_class][0]  # Use canonical prompt
        token_ids = torch.tensor(tokenize(prompt_text), dtype=torch.long).unsqueeze(0).to(device)

        image_files = sorted([
            f for f in os.listdir(ds_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            and not f.startswith('_') and f != 'masks'
        ])

        for img_file in image_files:
            img_path = os.path.join(ds_dir, img_file)
            orig_img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = orig_img.size

            # Transform for model
            img_np = np.array(orig_img)
            img_t, _ = transform(img_np, np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8))
            img_t = img_t.unsqueeze(0).to(device)

            with torch.no_grad(), autocast('cuda', enabled=use_amp):
                logits = model(img_t, token_ids)
                probs = torch.sigmoid(logits)

            # Resize back to original size
            mask = F.interpolate(probs, size=(orig_h, orig_w), mode='bilinear',
                                 align_corners=False)
            mask = (mask > 0.5).byte().squeeze().cpu().numpy() * 255

            # Save with naming convention: {image_id}__segment_{class}.png
            image_id = os.path.splitext(img_file)[0]
            label = prompt_class_to_label[prompt_class]
            out_name = f"{image_id}__segment_{label}.png"
            Image.fromarray(mask, mode='L').save(os.path.join(output_dir, out_name))
            count += 1

    print(f"Generated {count} prediction masks in {output_dir}")
    return count


def generate_visual_comparisons(model, cfg, device, output_dir, split="valid",
                                n_samples=8, use_amp=True):
    """Generate side-by-side comparison images: Original | GT | Prediction."""
    os.makedirs(output_dir, exist_ok=True)

    data_root = cfg["data"]["data_root"]
    image_size = cfg["data"]["image_size"]
    transform = get_val_transforms(image_size)

    datasets_config = [
        {"dir": "drywall_taping", "prompt_class": "taping"},
        {"dir": "cracks", "prompt_class": "crack"},
    ]

    samples_per_class = n_samples // 2

    for ds_cfg in datasets_config:
        ds_dir = os.path.join(data_root, ds_cfg["dir"], split)
        mask_dir = os.path.join(ds_dir, "masks")
        if not os.path.exists(ds_dir) or not os.path.exists(mask_dir):
            continue

        prompt_class = ds_cfg["prompt_class"]
        prompt_text = PROMPT_POOL[prompt_class][0]
        token_ids = torch.tensor(tokenize(prompt_text), dtype=torch.long).unsqueeze(0).to(device)

        image_files = sorted([
            f for f in os.listdir(ds_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            and not f.startswith('_') and f != 'masks'
        ])

        # Pick evenly spaced samples
        indices = np.linspace(0, len(image_files) - 1, samples_per_class, dtype=int)

        for i, idx in enumerate(indices):
            img_file = image_files[idx]
            img_path = os.path.join(ds_dir, img_file)
            stem = os.path.splitext(img_file)[0]
            gt_path = os.path.join(mask_dir, f"{stem}.png")

            if not os.path.exists(gt_path):
                continue

            # Load original and GT
            orig = Image.open(img_path).convert("RGB")
            gt = Image.open(gt_path).convert("L")

            # Run inference
            img_np = np.array(orig)
            img_t, _ = transform(img_np, np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8))
            img_t = img_t.unsqueeze(0).to(device)

            with torch.no_grad(), autocast('cuda', enabled=use_amp):
                logits = model(img_t, token_ids)
                probs = torch.sigmoid(logits)

            pred = F.interpolate(probs, size=(orig.size[1], orig.size[0]),
                                 mode='bilinear', align_corners=False)
            pred_mask = (pred > 0.5).byte().squeeze().cpu().numpy() * 255

            # Create side-by-side: Original | GT | Prediction
            w, h = orig.size
            canvas = Image.new("RGB", (w * 3, h))

            canvas.paste(orig, (0, 0))

            gt_rgb = Image.merge("RGB", [gt, gt, gt])
            canvas.paste(gt_rgb, (w, 0))

            pred_pil = Image.fromarray(pred_mask, mode='L')
            pred_rgb = Image.merge("RGB", [pred_pil, pred_pil, pred_pil])
            canvas.paste(pred_rgb, (w * 2, 0))

            out_path = os.path.join(output_dir, f"{prompt_class}_{i:03d}_comparison.png")
            canvas.save(out_path)

    print(f"Saved visual comparisons to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PromptSeg-Lite")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint (default: checkpoints/best_model.pt)")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--output-dir", default="report/predictions")
    parser.add_argument("--no-masks", action="store_true", help="Skip mask generation")
    parser.add_argument("--no-visuals", action="store_true", help="Skip visual comparisons")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.get("training", {}).get("amp", True)

    # Find best checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(cfg.get("checkpoint_dir", "checkpoints"), "best_model.pt")
    if not os.path.exists(ckpt_path):
        # Fall back to latest *_best.pt file
        ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
        best_files = sorted([
            f for f in os.listdir(ckpt_dir) if f.endswith("_best.pt")
        ])
        if best_files:
            ckpt_path = os.path.join(ckpt_dir, best_files[-1])
        else:
            ckpt_path = os.path.join(ckpt_dir, "latest.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    print("=" * 60)
    print("PromptSeg-Lite Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")

    # Load model
    model = load_model(ckpt_path, cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Create dataloaders for metrics
    _, val_loader = create_dataloaders(cfg)
    if args.split == "train":
        eval_loader = create_dataloaders(cfg)[0]
    else:
        eval_loader = val_loader

    # Run evaluation
    print("\nComputing metrics...")
    metrics = evaluate_dataset(model, eval_loader, device, use_amp)

    print("\n" + "-" * 40)
    print("EVALUATION RESULTS")
    print("-" * 40)
    print(f"  Overall Dice:    {metrics.get('dice_all', 0):.4f}")
    print(f"  Overall IoU:     {metrics.get('iou_all', 0):.4f}")
    print(f"  mIoU:            {metrics.get('miou', 0):.4f}")
    print(f"  Taping Dice:     {metrics.get('dice_taping', 0):.4f}")
    print(f"  Taping IoU:      {metrics.get('iou_taping', 0):.4f}")
    print(f"  Crack Dice:      {metrics.get('dice_crack', 0):.4f}")
    print(f"  Crack IoU:       {metrics.get('iou_crack', 0):.4f}")
    print(f"  Avg Latency:     {metrics.get('avg_latency_ms', 0):.2f} ms/image")
    print(f"  Throughput:      {metrics.get('throughput_fps', 0):.1f} FPS")
    print("-" * 40)

    # Save metrics
    os.makedirs("report", exist_ok=True)
    metrics_path = "report/evaluation_metrics.json"
    # Ensure serializable
    serializable = {k: float(v) for k, v in metrics.items()}
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Generate prediction masks
    if not args.no_masks:
        print("\nGenerating prediction masks...")
        generate_prediction_masks(model, cfg, device, args.output_dir,
                                  args.split, use_amp)

    # Generate visual comparisons
    if not args.no_visuals:
        print("\nGenerating visual comparisons...")
        generate_visual_comparisons(model, cfg, device,
                                    "report/figures/comparisons",
                                    args.split, n_samples=8, use_amp=use_amp)

    # Inference speed benchmark
    print("\nRunning speed benchmark (100 iterations)...")
    img_size = cfg["data"]["image_size"]
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    dummy_tokens = torch.ones(1, MAX_PROMPT_LEN, dtype=torch.long, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad(), autocast('cuda', enabled=use_amp):
            _ = model(dummy, dummy_tokens)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad(), autocast('cuda', enabled=use_amp):
            _ = model(dummy, dummy_tokens)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms_per_image = elapsed / 100 * 1000
    print(f"  Inference speed: {ms_per_image:.2f} ms/image ({1000/ms_per_image:.0f} FPS)")

    serializable["benchmark_ms_per_image"] = ms_per_image
    serializable["benchmark_fps"] = 1000 / ms_per_image
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
