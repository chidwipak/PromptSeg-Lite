"""PromptSegDataset: PyTorch Dataset for text-conditioned segmentation."""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

from src.data.prompt_pool import PROMPT_POOL, tokenize, MAX_PROMPT_LEN
from src.data.transforms import get_train_transforms, get_val_transforms


class PromptSegDataset(Dataset):
    """
    Returns: (image, mask, token_ids, prompt_class_idx)
    - image: (3, H, W) float32, normalized
    - mask: (1, H, W) float32, values {0.0, 1.0}
    - token_ids: (max_len,) int64, character-level token IDs
    - prompt_class_idx: int, 0=taping, 1=crack
    """

    CLASS_TO_IDX = {"taping": 0, "crack": 1}

    def __init__(self, data_root, datasets_config, split="train",
                 image_size=256, transform=None):
        """
        Args:
            data_root: Root data directory
            datasets_config: list of dicts with keys:
                - dir: dataset directory name under data_root
                - prompt_class: "taping" or "crack"
            split: "train" or "valid" or "test"
            image_size: target image size
            transform: augmentation pipeline
        """
        self.data_root = data_root
        self.image_size = image_size
        self.transform = transform
        self.split = split

        self.samples = []  # list of (image_path, mask_path, prompt_class)

        for ds_cfg in datasets_config:
            ds_dir = os.path.join(data_root, ds_cfg["dir"], split)
            mask_dir = os.path.join(ds_dir, "masks")
            prompt_class = ds_cfg["prompt_class"]

            if not os.path.exists(ds_dir):
                print(f"[WARN] Split directory not found: {ds_dir}")
                continue
            if not os.path.exists(mask_dir):
                print(f"[WARN] Mask directory not found: {mask_dir}")
                continue

            # Collect image-mask pairs
            image_files = sorted([
                f for f in os.listdir(ds_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                and not f.startswith('_')
                and f != 'masks'
            ])

            for img_file in image_files:
                stem = os.path.splitext(img_file)[0]
                mask_file = f"{stem}.png"
                mask_path = os.path.join(mask_dir, mask_file)

                if os.path.exists(mask_path):
                    self.samples.append((
                        os.path.join(ds_dir, img_file),
                        mask_path,
                        prompt_class,
                    ))

        print(f"[Dataset] {split}: {len(self.samples)} samples loaded")
        # Count per class
        class_counts = {}
        for _, _, pc in self.samples:
            class_counts[pc] = class_counts.get(pc, 0) + 1
        for cls, cnt in class_counts.items():
            print(f"  {cls}: {cnt} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, prompt_class = self.samples[idx]

        # Load image (RGB)
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load mask (single channel)
        mask = np.array(Image.open(mask_path).convert("L"))

        # Apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)

        # Sample a random prompt from the pool
        prompt_text = random.choice(PROMPT_POOL[prompt_class])
        token_ids = torch.tensor(tokenize(prompt_text), dtype=torch.long)

        class_idx = self.CLASS_TO_IDX[prompt_class]

        return image, mask, token_ids, class_idx

    def get_sampler_weights(self):
        """Compute per-sample weights for balanced sampling across classes."""
        class_counts = {}
        for _, _, pc in self.samples:
            class_counts[pc] = class_counts.get(pc, 0) + 1

        total = len(self.samples)
        n_classes = len(class_counts)
        class_weight = {cls: total / (n_classes * cnt)
                        for cls, cnt in class_counts.items()}

        weights = [class_weight[pc] for _, _, pc in self.samples]
        return weights


def create_dataloaders(cfg):
    """Create train and validation dataloaders with balanced sampling."""
    data_root = cfg["data"]["data_root"]
    image_size = cfg["data"]["image_size"]
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    datasets_config = [
        {"dir": "drywall_taping", "prompt_class": "taping"},
        {"dir": "cracks", "prompt_class": "crack"},
    ]

    # Transforms
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    # Datasets
    train_dataset = PromptSegDataset(
        data_root, datasets_config, split="train",
        image_size=image_size, transform=train_transform,
    )
    val_dataset = PromptSegDataset(
        data_root, datasets_config, split="valid",
        image_size=image_size, transform=val_transform,
    )

    # Balanced sampler for training (handles class imbalance)
    train_weights = train_dataset.get_sampler_weights()
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
