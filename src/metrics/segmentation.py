"""Segmentation metrics: IoU and Dice score."""

import torch


def compute_iou(pred, target, threshold=0.5, smooth=1e-6):
    """Compute Intersection over Union.
    
    Args:
        pred: (B, 1, H, W) probabilities or logits
        target: (B, 1, H, W) binary ground truth {0, 1}
        threshold: binarization threshold
    Returns:
        mean IoU across batch
    """
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = torch.sigmoid(pred)
    
    pred_binary = (pred > threshold).float()
    
    pred_flat = pred_binary.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def compute_dice(pred, target, threshold=0.5, smooth=1e-6):
    """Compute Dice score (F1).
    
    Args:
        pred: (B, 1, H, W) probabilities or logits
        target: (B, 1, H, W) binary ground truth {0, 1}
        threshold: binarization threshold
    Returns:
        mean Dice score across batch
    """
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = torch.sigmoid(pred)
    
    pred_binary = (pred > threshold).float()
    
    pred_flat = pred_binary.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    total = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    
    dice = (2.0 * intersection + smooth) / (total + smooth)
    return dice.mean().item()


class MetricTracker:
    """Track per-class and overall metrics during evaluation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.iou_sum = {"taping": 0.0, "crack": 0.0, "all": 0.0}
        self.dice_sum = {"taping": 0.0, "crack": 0.0, "all": 0.0}
        self.count = {"taping": 0, "crack": 0, "all": 0}

    def update(self, pred, target, class_idx):
        """Update with a batch of predictions.
        
        Args:
            pred: (B, 1, H, W) logits
            target: (B, 1, H, W) binary GT
            class_idx: (B,) class indices (0=taping, 1=crack)
        """
        iou = compute_iou(pred, target)
        dice = compute_dice(pred, target)
        
        self.iou_sum["all"] += iou * pred.size(0)
        self.dice_sum["all"] += dice * pred.size(0)
        self.count["all"] += pred.size(0)

        class_names = {0: "taping", 1: "crack"}
        for cls_val, cls_name in class_names.items():
            mask = (class_idx == cls_val)
            if mask.any():
                cls_pred = pred[mask]
                cls_target = target[mask]
                cls_iou = compute_iou(cls_pred, cls_target)
                cls_dice = compute_dice(cls_pred, cls_target)
                n = mask.sum().item()
                self.iou_sum[cls_name] += cls_iou * n
                self.dice_sum[cls_name] += cls_dice * n
                self.count[cls_name] += n

    def compute(self):
        """Compute final averaged metrics."""
        results = {}
        for key in ["taping", "crack", "all"]:
            n = self.count[key]
            if n > 0:
                results[f"iou_{key}"] = self.iou_sum[key] / n
                results[f"dice_{key}"] = self.dice_sum[key] / n
            else:
                results[f"iou_{key}"] = 0.0
                results[f"dice_{key}"] = 0.0

        # mIoU = mean of per-class IoU
        class_ious = []
        for key in ["taping", "crack"]:
            if self.count[key] > 0:
                class_ious.append(self.iou_sum[key] / self.count[key])
        results["miou"] = sum(class_ious) / len(class_ious) if class_ious else 0.0
        return results
