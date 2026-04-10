#!/usr/bin/env python3
"""PromptSeg-Lite Training Script with AMP, hooks, and comprehensive logging."""

import os
import sys
import time
import random
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import create_dataloaders
from src.model.promptseg import PromptSegLite
from src.losses.dice_bce import DiceBCELoss
from src.losses.compound_v2 import CompoundV2Loss, DeepSupervisionLoss
from src.metrics.segmentation import MetricTracker, compute_iou, compute_dice
from src.utils.logger import MetricsLogger
from src.utils.hooks import FeatureGradientCapture, generate_layer_specs
from src.utils.visualization import (
    visualize_feature_maps, visualize_predictions, plot_training_curves
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path="config/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                    device, use_amp=True):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n_batches = 0

    for images, masks, token_ids, class_idx in train_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        token_ids = token_ids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=use_amp):
            output = model(images, token_ids)
            loss = criterion(output, masks)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[WARN] NaN/Inf loss detected, skipping batch")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        # Extract main logits for metrics (handle deep supervision dict)
        with torch.no_grad():
            logits = output['main'] if isinstance(output, dict) else output
            total_dice += compute_dice(logits, masks)
            total_iou += compute_iou(logits, masks)
        n_batches += 1

    if n_batches == 0:
        return float('nan'), 0.0, 0.0

    return total_loss / n_batches, total_dice / n_batches, total_iou / n_batches


@torch.no_grad()
def validate(model, val_loader, criterion, device, use_amp=True):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    tracker = MetricTracker()

    for images, masks, token_ids, class_idx in val_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        token_ids = token_ids.to(device, non_blocking=True)
        class_idx = class_idx.to(device, non_blocking=True)

        with autocast('cuda', enabled=use_amp):
            output = model(images, token_ids)
            # In eval mode, decoder returns tensor (not dict)
            logits = output['main'] if isinstance(output, dict) else output
            # Use base loss for validation (no deep supervision)
            if hasattr(criterion, 'base_loss'):
                loss = criterion.base_loss(logits, masks)
            else:
                loss = criterion(logits, masks)

        total_loss += loss.item()
        tracker.update(logits, masks, class_idx)
        n_batches += 1

    metrics = tracker.compute()
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def save_checkpoint(model, optimizer, scaler, scheduler, epoch, metrics,
                    path, is_best=False):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, path)
    if is_best:
        best_path = path.replace(".pt", "_best.pt")
        torch.save(state, best_path)
        # Also save as best_model.pt for easy access
        best_model_path = os.path.join(os.path.dirname(path), "best_model.pt")
        torch.save(state, best_model_path)


def main():
    parser = argparse.ArgumentParser(description="Train PromptSeg-Lite")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["training"].get("amp", True) and device.type == "cuda"

    print("=" * 70)
    print("PromptSeg-Lite Training")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"AMP: {use_amp}")
    print(f"Seed: {cfg['seed']}")
    print()

    # Data
    print("Loading datasets...")
    train_loader, val_loader = create_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Batch size: {cfg['data']['batch_size']}")
    print()

    # Model
    print("Building model...")
    model = PromptSegLite(cfg.get("model", {})).to(device)
    total_params, trainable_params = model.count_parameters()
    print(f"Total parameters: {total_params:,} ({total_params * 4 / 1e6:.1f} MB FP32)")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Loss, optimizer, scheduler
    loss_cfg = cfg["training"]["loss"]
    loss_type = loss_cfg.get("type", "dice_bce")

    if loss_type == "compound_v2":
        base_loss = CompoundV2Loss(
            ft_weight=loss_cfg.get("ft_weight", 1.0),
            boundary_weight=loss_cfg.get("boundary_weight", 0.5),
            ohem_weight=loss_cfg.get("ohem_weight", 0.5),
            ft_alpha=loss_cfg.get("ft_alpha", 0.7),
            ft_beta=loss_cfg.get("ft_beta", 0.3),
            ft_gamma=loss_cfg.get("ft_gamma", 0.75),
            ohem_ratio=loss_cfg.get("ohem_ratio", 0.25),
        )
        aux_weights = loss_cfg.get("aux_weights", [0.4, 0.2])
        criterion = DeepSupervisionLoss(base_loss, aux_weights=tuple(aux_weights))
        print(f"Loss: CompoundV2 (FT:{loss_cfg.get('ft_weight',1.0)}, "
              f"Boundary:{loss_cfg.get('boundary_weight',0.5)}, "
              f"OHEM:{loss_cfg.get('ohem_weight',0.5)}) + DeepSupervision")
    else:
        criterion = DiceBCELoss(
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            bce_weight=loss_cfg.get("bce_weight", 1.0),
        )
        print(f"Loss: DiceBCE (Dice:{loss_cfg.get('dice_weight',1.0)}, "
              f"BCE:{loss_cfg.get('bce_weight',1.0)})")

    train_cfg = cfg["training"]
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Cosine annealing with warmup
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    total_epochs = train_cfg["epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            min_lr = train_cfg.get("min_lr", 1e-6)
            base_lr = train_cfg["learning_rate"]
            return max(min_lr / base_lr,
                       0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler(enabled=use_amp)

    # Logger
    logger = MetricsLogger(
        log_dir=cfg.get("log_dir", "runs"),
        experiment_name="promptseg_lite",
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_dice = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_dice = ckpt.get("metrics", {}).get("dice_all", 0.0)
        print(f"Resumed at epoch {start_epoch}, best val dice: {best_val_dice:.4f}")

    # DL Course: Layer specification table (run once)
    print("Generating layer specifications (DL Course Slides 3-5)...")
    os.makedirs("report/figures", exist_ok=True)
    try:
        img_sz = cfg["data"]["image_size"]
        sample_img = torch.randn(1, 3, img_sz, img_sz).to(device)
        sample_tok = torch.ones(1, 25, dtype=torch.long).to(device)
        specs = generate_layer_specs(model, sample_img, sample_tok)
        with open("report/layer_specs.json", 'w') as f:
            json.dump(specs, f, indent=2, default=str)
        print(f"  Saved {len(specs)} layer specs to report/layer_specs.json")
    except Exception as e:
        print(f"  [WARN] Layer spec generation failed: {e}")

    # DL Course: Register feature/gradient hooks (Slide 6)
    capture = FeatureGradientCapture()
    hook_layers = [
        "vision_encoder.stage1.convs.0.pw",  # Shallow encoder features
        "decoder.dec3.conv.pw",               # Deep decoder features
    ]
    capture.register(model, hook_layers)

    # Checkpoint dir
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Early stopping
    es_cfg = train_cfg.get("early_stopping", {})
    patience = es_cfg.get("patience", 25)
    no_improve_count = 0

    # Training loop
    print("=" * 70)
    print(f"Starting training for {total_epochs} epochs")
    print("=" * 70)
    logger.log_message(f"Training started: {total_epochs} epochs, "
                       f"batch_size={cfg['data']['batch_size']}, "
                       f"lr={train_cfg['learning_rate']}, "
                       f"AMP={use_amp}")

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()

        # Train
        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, use_amp=use_amp
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, use_amp=use_amp)

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - t0

        # Log metrics
        metrics = {
            "train_loss": train_loss,
            "train_dice": train_dice,
            "train_iou": train_iou,
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice_all"],
            "val_iou": val_metrics["iou_all"],
            "val_miou": val_metrics["miou"],
            "val_dice_taping": val_metrics["dice_taping"],
            "val_dice_crack": val_metrics["dice_crack"],
            "val_iou_taping": val_metrics["iou_taping"],
            "val_iou_crack": val_metrics["iou_crack"],
            "lr": current_lr,
            "epoch_time": epoch_time,
        }
        logger.log_epoch(epoch, metrics)

        # Check for best model
        val_dice = val_metrics["dice_all"]
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Save checkpoint
        if (epoch + 1) % 10 == 0 or is_best:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
            save_checkpoint(model, optimizer, scaler, scheduler, epoch,
                            val_metrics, ckpt_path, is_best=is_best)

        # Always save latest
        save_checkpoint(model, optimizer, scaler, scheduler, epoch,
                        val_metrics,
                        os.path.join(ckpt_dir, "latest.pt"))

        # Print progress
        if (epoch + 1) % 5 == 0 or is_best:
            print(f"Epoch {epoch:04d}/{total_epochs} | "
                  f"Loss: {train_loss:.4f}/{val_metrics['loss']:.4f} | "
                  f"Dice: {train_dice:.4f}/{val_dice:.4f} | "
                  f"mIoU: {val_metrics['miou']:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.1f}s"
                  f"{' *BEST*' if is_best else ''}")

        # Early stopping
        if no_improve_count >= patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            logger.log_message(f"Early stopping at epoch {epoch}")
            break

        # NaN detection
        if np.isnan(train_loss):
            print(f"\n[CRITICAL] NaN loss at epoch {epoch}! Stopping.")
            logger.log_message(f"CRITICAL: NaN loss at epoch {epoch}")
            break

    # Save training history
    logger.save_history()

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Training curves (Slide 8)
    try:
        plot_training_curves(logger.history, "report/figures/training_curves.png")
        print("  Saved training curves")
    except Exception as e:
        print(f"  [WARN] Training curves failed: {e}")

    # Feature maps and gradients (Slide 6)
    try:
        model.eval()
        # Get one validation batch
        val_iter = iter(val_loader)
        images, masks, token_ids, class_idx = next(val_iter)
        images = images.to(device)
        masks = masks.to(device)
        token_ids = token_ids.to(device)

        # Forward + backward to capture features and gradients
        model.train()
        output = model(images, token_ids)
        loss = criterion(output, masks)
        loss.backward()

        for layer_name in hook_layers:
            visualize_feature_maps(capture, layer_name, "report/figures")
        print("  Saved feature map visualizations")
    except Exception as e:
        print(f"  [WARN] Feature map visualization failed: {e}")

    # Prediction examples (Report)
    try:
        model.eval()
        with torch.no_grad():
            val_iter = iter(val_loader)
            images, masks, token_ids, class_idx = next(val_iter)
            images = images.to(device)
            token_ids = token_ids.to(device)
            with autocast('cuda', enabled=use_amp):
                output = model(images, token_ids)
                logits = output['main'] if isinstance(output, dict) else output
            preds = (torch.sigmoid(logits) > 0.5).float()

        visualize_predictions(images, masks, preds,
                              "report/figures/prediction_examples.png",
                              num_samples=4)
        print("  Saved prediction examples")
    except Exception as e:
        print(f"  [WARN] Prediction visualization failed: {e}")

    capture.remove_hooks()
    logger.close()

    print("\n" + "=" * 70)
    print(f"Training complete. Best val Dice: {best_val_dice:.4f}")
    print(f"Checkpoints saved to: {ckpt_dir}/")
    print(f"Logs saved to: {cfg.get('log_dir', 'runs')}/")
    print(f"Figures saved to: report/figures/")
    print("=" * 70)

    return best_val_dice


if __name__ == "__main__":
    main()
