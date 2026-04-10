"""Training metrics logger with TensorBoard and file logging."""

import os
import json
import time
import logging


class MetricsLogger:
    """Track and log all training metrics."""

    def __init__(self, log_dir="runs", experiment_name="promptseg"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_dice": [], "val_dice": [],
            "train_iou": [], "val_iou": [],
            "val_miou": [],
            "val_dice_taping": [], "val_dice_crack": [],
            "val_iou_taping": [], "val_iou_crack": [],
            "lr": [],
            "epoch_time": [],
        }

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(log_dir, experiment_name))
        except ImportError:
            self.writer = None

        # File logger
        self.file_logger = logging.getLogger("training")
        self.file_logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        self.file_logger.addHandler(fh)

        # Also log to stdout
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s | %(message)s",
                                          datefmt="%H:%M:%S"))
        self.file_logger.addHandler(sh)

    def log_epoch(self, epoch, metrics_dict):
        """Log metrics for one epoch."""
        for key, value in metrics_dict.items():
            if key in self.history:
                self.history[key].append(value)
            if self.writer:
                self.writer.add_scalar(key, value, epoch)

        # Log to file
        msg = f"Epoch {epoch:04d}"
        for key, value in sorted(metrics_dict.items()):
            if isinstance(value, float):
                msg += f" | {key}: {value:.4f}"
            else:
                msg += f" | {key}: {value}"
        self.file_logger.info(msg)

    def log_message(self, msg):
        self.file_logger.info(msg)

    def save_history(self, path=None):
        """Save training history to JSON."""
        if path is None:
            path = os.path.join(self.log_dir, "training_history.json")
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def close(self):
        if self.writer:
            self.writer.close()
