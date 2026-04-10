#!/usr/bin/env python3
"""Single-image inference for PromptSeg-Lite."""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from PIL import Image
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.prompt_pool import PROMPT_POOL, tokenize, MAX_PROMPT_LEN
from src.data.transforms import get_val_transforms
from src.model.promptseg import PromptSegLite


def load_model(checkpoint_path, cfg, device):
    model = PromptSegLite(cfg.get("model", {})).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def predict(model, image_path, prompt_text, cfg, device, threshold=0.5):
    """Run inference on a single image with a text prompt."""
    image_size = cfg["data"]["image_size"]
    use_amp = cfg.get("training", {}).get("amp", True)
    transform = get_val_transforms(image_size)

    orig = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig.size

    img_np = np.array(orig)
    img_t, _ = transform(img_np, np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8))
    img_t = img_t.unsqueeze(0).to(device)

    token_ids = torch.tensor(tokenize(prompt_text), dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad(), autocast('cuda', enabled=use_amp):
        logits = model(img_t, token_ids)
        probs = torch.sigmoid(logits)

    # Resize to original
    mask = F.interpolate(probs, size=(orig_h, orig_w), mode='bilinear',
                         align_corners=False)
    mask = (mask > threshold).byte().squeeze().cpu().numpy() * 255

    return mask


def main():
    parser = argparse.ArgumentParser(description="PromptSeg-Lite Single-Image Inference")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--prompt", required=True,
                        help="Text prompt, e.g. 'segment crack' or 'segment taping area'")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None, help="Output mask path")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overlay", action="store_true",
                        help="Also save overlay visualization")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint
    if ckpt_path is None:
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

    model = load_model(ckpt_path, cfg, device)

    mask = predict(model, args.image, args.prompt, cfg, device, args.threshold)

    # Output path
    if args.output:
        out_path = args.output
    else:
        stem = os.path.splitext(os.path.basename(args.image))[0]
        prompt_label = args.prompt.replace(" ", "_")
        out_path = f"{stem}__{prompt_label}.png"

    Image.fromarray(mask, mode='L').save(out_path)
    print(f"Saved mask to {out_path}")

    if args.overlay:
        orig = Image.open(args.image).convert("RGB")
        orig_np = np.array(orig)
        mask_bool = mask > 127

        overlay = orig_np.copy()
        overlay[mask_bool] = (
            overlay[mask_bool] * 0.5 +
            np.array([0, 255, 0], dtype=np.uint8) * 0.5
        ).astype(np.uint8)

        overlay_path = out_path.replace(".png", "_overlay.png")
        Image.fromarray(overlay).save(overlay_path)
        print(f"Saved overlay to {overlay_path}")


if __name__ == "__main__":
    main()
