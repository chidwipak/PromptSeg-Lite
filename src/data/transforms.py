"""Joint image+mask augmentation transforms for segmentation training."""

import random
import numpy as np
import torch
import cv2


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class Resize:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image, mask):
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        return image, mask


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        return image, mask


class RandomVerticalFlip:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        return image, mask


class RandomRotation:
    def __init__(self, degrees=15):
        self.degrees = degrees

    def __call__(self, image, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return image, mask


class ColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, mask):
        # Brightness
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        # Contrast
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = image.mean()
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        # Saturation
        if self.saturation > 0:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        # Hue
        if self.hue > 0:
            shift = random.uniform(-self.hue * 180, self.hue * 180)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return image, mask


class GaussianBlur:
    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), p=0.3):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            sigma = random.uniform(*self.sigma)
            image = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), sigma)
        return image, mask


class GaussianNoise:
    def __init__(self, std=0.02, p=0.2):
        self.std = std
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            noise = np.random.randn(*image.shape) * self.std * 255
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return image, mask


class Normalize:
    """Normalize to [0,1] then apply ImageNet-style normalization."""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, mask):
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image, mask


class ToTensor:
    """Convert numpy arrays to PyTorch tensors."""
    def __call__(self, image, mask):
        # image: (H, W, 3) -> (3, H, W)
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        # mask: (H, W) -> (1, H, W), normalized to [0, 1]
        if mask.ndim == 2:
            mask = mask[np.newaxis, :, :]
        mask = torch.from_numpy(mask.copy()).float() / 255.0
        return image, mask


def get_train_transforms(image_size=256, cfg=None):
    """Get training augmentation pipeline."""
    transforms = [
        Resize(image_size),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.3),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        GaussianBlur(p=0.3),
        GaussianNoise(p=0.2),
        Normalize(),
        ToTensor(),
    ]
    return Compose(transforms)


def get_val_transforms(image_size=256):
    """Get validation/test transform pipeline (deterministic)."""
    transforms = [
        Resize(image_size),
        Normalize(),
        ToTensor(),
    ]
    return Compose(transforms)
