#!/usr/bin/env python3
"""Generate binary masks from COCO bounding box annotations.
Since both datasets have bbox-only annotations, we create filled rectangle masks.
For drywall joints (linear structures), we use the bbox directly.
For cracks, we also use bbox as ground truth regions.
Output: {split}/masks/{image_filename_stem}.png  (single-channel, {0, 255})
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from collections import defaultdict


def generate_masks_for_split(data_dir, split, dataset_name):
    """Generate binary masks for all images in a split."""
    split_dir = os.path.join(data_dir, split)
    ann_path = os.path.join(split_dir, "_annotations.coco.json")
    
    if not os.path.exists(ann_path):
        print(f"  [SKIP] No annotations found for {dataset_name}/{split}")
        return 0
    
    with open(ann_path, 'r') as f:
        coco = json.load(f)
    
    # Build image_id -> annotations mapping
    img_id_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_id_to_anns[ann["image_id"]].append(ann)
    
    # Build image_id -> image info
    img_id_to_info = {img["id"]: img for img in coco["images"]}
    
    # Create masks directory
    mask_dir = os.path.join(split_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    
    count = 0
    no_ann_count = 0
    
    for img_info in coco["images"]:
        img_id = img_info["id"]
        img_w = img_info["width"]
        img_h = img_info["height"]
        img_filename = img_info["file_name"]
        
        # Create blank mask
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        anns = img_id_to_anns.get(img_id, [])
        
        if not anns:
            no_ann_count += 1
        
        for ann in anns:
            # Try segmentation first (polygon)
            seg = ann.get("segmentation", [])
            if seg and isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list) and len(seg[0]) >= 6:
                # Has polygon segmentation - rasterize it
                from PIL import ImageDraw
                poly_mask = Image.new('L', (img_w, img_h), 0)
                draw = ImageDraw.Draw(poly_mask)
                for polygon in seg:
                    if len(polygon) >= 6:
                        coords = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                        draw.polygon(coords, fill=255)
                mask = np.maximum(mask, np.array(poly_mask))
            elif ann.get("bbox"):
                # Fallback: use bounding box
                x, y, w, h = [int(round(v)) for v in ann["bbox"]]
                x = max(0, x)
                y = max(0, y)
                x2 = min(img_w, x + w)
                y2 = min(img_h, y + h)
                mask[y:y2, x:x2] = 255
        
        # Save mask
        stem = os.path.splitext(img_filename)[0]
        mask_path = os.path.join(mask_dir, f"{stem}.png")
        Image.fromarray(mask).save(mask_path)
        count += 1
    
    print(f"  [OK] {dataset_name}/{split}: Generated {count} masks "
          f"({no_ann_count} images had no annotations -> empty masks)")
    return count


def main():
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    datasets = {
        "drywall_taping": "taping",
        "cracks": "crack",
    }
    
    print("=" * 60)
    print("MASK GENERATION FROM COCO ANNOTATIONS")
    print("=" * 60)
    
    total = 0
    for ds_dir, prompt_class in datasets.items():
        ds_path = os.path.join(data_root, ds_dir)
        if not os.path.exists(ds_path):
            print(f"[SKIP] Dataset not found: {ds_path}")
            continue
        
        print(f"\n--- {ds_dir} (class: {prompt_class}) ---")
        for split in ["train", "valid", "test"]:
            split_path = os.path.join(ds_path, split)
            if os.path.exists(split_path):
                n = generate_masks_for_split(ds_path, split, ds_dir)
                total += n
    
    print(f"\n{'='*60}")
    print(f"TOTAL: Generated {total} masks")
    print(f"{'='*60}")
    
    # Verify a few masks
    print("\nVerification (sample masks):")
    for ds_dir in datasets:
        mask_dir = os.path.join(data_root, ds_dir, "train", "masks")
        if os.path.exists(mask_dir):
            masks = sorted(os.listdir(mask_dir))[:3]
            for m in masks:
                mask = np.array(Image.open(os.path.join(mask_dir, m)))
                fg_pct = (mask > 0).sum() / mask.size * 100
                print(f"  {ds_dir}/train/masks/{m}: shape={mask.shape}, "
                      f"unique={np.unique(mask).tolist()}, fg={fg_pct:.1f}%")


if __name__ == "__main__":
    main()
