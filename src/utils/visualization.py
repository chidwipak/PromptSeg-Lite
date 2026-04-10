"""Visualization utilities for feature maps, gradients, and predictions."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def visualize_feature_maps(capture, layer_name, save_dir, num_channels=8):
    """Plot activation and gradient channels for a layer.
    
    For Slide 6 of DL course presentation.
    """
    os.makedirs(save_dir, exist_ok=True)

    act = capture.activations.get(layer_name)
    grad = capture.gradients.get(layer_name)

    if act is None:
        print(f"[WARN] No activation captured for {layer_name}")
        return

    act = act[0]  # First sample in batch (C, H, W)
    n = min(num_channels, act.shape[0])

    rows = 2 if grad is not None else 1
    fig, axes = plt.subplots(rows, n, figsize=(2.5 * n, 2.5 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        axes[0, i].imshow(act[i].numpy(), cmap='viridis')
        axes[0, i].set_title(f'Act ch{i}', fontsize=8)
        axes[0, i].axis('off')

    if grad is not None:
        grad = grad[0]  # (C, H, W)
        for i in range(n):
            axes[1, i].imshow(grad[i].numpy(), cmap='coolwarm')
            axes[1, i].set_title(f'Grad ch{i}', fontsize=8)
            axes[1, i].axis('off')

    fig.suptitle(f'Layer: {layer_name}', fontsize=12)
    plt.tight_layout()
    safe_name = layer_name.replace('.', '_')
    fig.savefig(os.path.join(save_dir, f'features_{safe_name}.png'), dpi=150)
    plt.close(fig)


def visualize_predictions(images, masks_gt, masks_pred, save_path,
                          num_samples=4, mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225)):
    """Plot original | GT mask | predicted mask comparisons.
    
    For report visual examples.
    """
    n = min(num_samples, images.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        # Denormalize image
        img = images[i].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
        img = img * np.array(std) + np.array(mean)
        img = np.clip(img, 0, 1)

        gt = masks_gt[i, 0].cpu().numpy()
        pred = masks_pred[i, 0].cpu().numpy()

        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_curves(history, save_path):
    """Plot training/validation loss, dice, and IoU curves.
    
    For Slide 8 of DL course presentation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    if history.get("train_loss") and history.get("val_loss"):
        axes[0].plot(history["train_loss"], label="Train Loss", color='#2196F3')
        axes[0].plot(history["val_loss"], label="Val Loss", color='#FF5722')
        axes[0].set_title("Loss Trajectory", fontsize=14)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Dice
    if history.get("train_dice") and history.get("val_dice"):
        axes[1].plot(history["train_dice"], label="Train Dice", color='#2196F3')
        axes[1].plot(history["val_dice"], label="Val Dice", color='#FF5722')
        if history.get("val_dice_taping"):
            axes[1].plot(history["val_dice_taping"], '--', label="Val Dice (Taping)", color='#4CAF50')
        if history.get("val_dice_crack"):
            axes[1].plot(history["val_dice_crack"], '--', label="Val Dice (Crack)", color='#9C27B0')
        axes[1].set_title("Dice Score", fontsize=14)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Dice")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # IoU
    if history.get("val_miou"):
        axes[2].plot(history["val_miou"], label="Val mIoU", color='#FF5722')
        if history.get("val_iou_taping"):
            axes[2].plot(history["val_iou_taping"], '--', label="Val IoU (Taping)", color='#4CAF50')
        if history.get("val_iou_crack"):
            axes[2].plot(history["val_iou_crack"], '--', label="Val IoU (Crack)", color='#9C27B0')
        axes[2].set_title("Mean IoU", fontsize=14)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("mIoU")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
