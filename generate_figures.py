"""Generate uniform figure sets for V1 and V2 models.

Each version gets the same 6 figures:
  01_training_curves.png   - Loss, Dice, mIoU convergence
  02_predictions.png       - Image | GT | Prediction grid (crack + taping)
  03_encoder_features.png  - Vision encoder feature maps at multiple stages
  04_text_conditioning.png - FiLM gamma/beta activations for crack vs taping
  05_prompt_validation.png - Same image, two prompts → two different masks
  06_metrics_summary.png   - Per-class IoU/Dice bar charts
"""

import os, re, sys, json, argparse, yaml
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from src.model.promptseg import PromptSegLite
from src.data.prompt_pool import tokenize, PROMPT_POOL
from src.data.transforms import get_val_transforms


def load_model(cfg, checkpoint_path, device):
    """Load a model from checkpoint, handling V1 (no ASPP/deep_sup) gracefully."""
    model = PromptSegLite(cfg['model'])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict']
    # Filter out unexpected keys (V1 checkpoint won't have ASPP/aux heads)
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state.items() if k in model_keys}
    missing = model_keys - set(filtered.keys())
    if missing:
        print(f"  Skipping {len(missing)} missing keys (expected for version mismatch)")
    model.load_state_dict(filtered, strict=False)
    model = model.to(device)
    model.eval()
    return model


def denormalize(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Convert normalized tensor to displayable numpy array."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    return np.clip(img, 0, 1)


# ─── Figure 1: Training Curves ───────────────────────────────────────────────

def generate_training_curves(history, save_path, version_label):
    """Generate 3-panel training curves: Loss, Dice, mIoU."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f'{version_label} — Training Curves', fontsize=14, fontweight='bold')

    epochs = range(1, len(history.get('train_loss', [])) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train', color='#1976D2', alpha=0.85)
    axes[0].plot(epochs, history['val_loss'], label='Validation', color='#D32F2F', alpha=0.85)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Convergence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dice
    axes[1].plot(epochs, history['train_dice'], label='Train', color='#1976D2', alpha=0.85)
    axes[1].plot(epochs, history['val_dice'], label='Validation', color='#D32F2F', alpha=0.85)
    if 'val_dice_taping' in history:
        axes[1].plot(epochs, history['val_dice_taping'], '--', label='Val Taping', color='#388E3C', alpha=0.7)
    if 'val_dice_crack' in history:
        axes[1].plot(epochs, history['val_dice_crack'], '--', label='Val Crack', color='#7B1FA2', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Dice Score')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # mIoU
    axes[2].plot(epochs, history['val_miou'], label='Val mIoU', color='#D32F2F', alpha=0.85)
    if 'val_iou_taping' in history:
        axes[2].plot(epochs, history['val_iou_taping'], '--', label='Val IoU Taping', color='#388E3C', alpha=0.7)
    if 'val_iou_crack' in history:
        axes[2].plot(epochs, history['val_iou_crack'], '--', label='Val IoU Crack', color='#7B1FA2', alpha=0.7)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU')
    axes[2].set_title('Mean IoU')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {save_path}')


# ─── Figure 2: Predictions ───────────────────────────────────────────────────

def generate_predictions(model, data_root, datasets_config, image_size, device, save_path, version_label):
    """Generate 2×3 prediction grid: 2 crack + 2 taping rows, each with Image|GT|Pred."""
    transform = get_val_transforms(image_size)

    samples = {'crack': [], 'taping': []}
    for ds_cfg in datasets_config:
        ds_dir = os.path.join(data_root, ds_cfg['dir'], 'valid')
        mask_dir = os.path.join(ds_dir, 'masks')
        pc = ds_cfg['prompt_class']
        imgs = sorted([f for f in os.listdir(ds_dir)
                       if f.lower().endswith(('.jpg', '.png')) and not f.startswith('_') and f != 'masks'])
        for img_file in imgs[:200]:
            stem = os.path.splitext(img_file)[0]
            mask_path = os.path.join(mask_dir, f'{stem}.png')
            if os.path.exists(mask_path):
                samples[pc].append((os.path.join(ds_dir, img_file), mask_path))

    # Pick good examples (not the first — pick from middle for variety)
    np.random.seed(42)
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    fig.suptitle(f'{version_label} — Prediction Examples', fontsize=14, fontweight='bold')

    row = 0
    for cls_name in ['taping', 'crack']:
        idxs = np.random.choice(len(samples[cls_name]), size=2, replace=False)
        prompt = PROMPT_POOL[cls_name][0]
        tokens = torch.tensor([tokenize(prompt)], dtype=torch.long).to(device)

        for idx in idxs:
            img_path, mask_path = samples[cls_name][idx]
            img_pil = Image.open(img_path).convert('RGB')
            gt_pil = Image.open(mask_path).convert('L')

            # Apply transforms
            img_np = np.array(img_pil.resize((image_size, image_size)))
            gt_np = np.array(gt_pil.resize((image_size, image_size), Image.NEAREST))
            img_t, gt_t = transform(img_np, gt_np)
            img_t = img_t.unsqueeze(0).to(device)

            with torch.no_grad():
                mask_pred, probs = model.predict(img_t, tokens)

            # Display
            img_display = denormalize(img_t[0])
            gt_display = gt_t.squeeze().numpy() / 255.0 if gt_t.squeeze().numpy().max() > 1 else gt_t.squeeze().numpy()
            pred_display = mask_pred[0, 0].cpu().numpy()

            axes[row, 0].imshow(img_display)
            axes[row, 0].set_title(f'Input ({cls_name})', fontsize=10)
            axes[row, 0].axis('off')

            axes[row, 1].imshow(gt_display, cmap='gray', vmin=0, vmax=1)
            axes[row, 1].set_title('Ground Truth', fontsize=10)
            axes[row, 1].axis('off')

            axes[row, 2].imshow(pred_display, cmap='gray', vmin=0, vmax=1)
            axes[row, 2].set_title('Prediction', fontsize=10)
            axes[row, 2].axis('off')
            row += 1

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {save_path}')


# ─── Figure 3: Vision Encoder Features ───────────────────────────────────────

def generate_encoder_features(model, data_root, datasets_config, image_size, device, save_path, version_label):
    """Visualize feature maps from vision encoder stages 1-4."""
    transform = get_val_transforms(image_size)

    # Get a crack sample
    ds_dir = os.path.join(data_root, datasets_config[1]['dir'], 'valid')
    imgs = sorted([f for f in os.listdir(ds_dir) if f.lower().endswith(('.jpg', '.png')) and f != 'masks'])
    img_path = os.path.join(ds_dir, imgs[10])  # pick a specific sample
    img_pil = Image.open(img_path).convert('RGB')
    img_np = np.array(img_pil.resize((image_size, image_size)))
    img_t, _ = transform(img_np, np.zeros((image_size, image_size), dtype=np.uint8))
    img_t = img_t.unsqueeze(0).to(device)

    prompt = "segment crack"
    tokens = torch.tensor([tokenize(prompt)], dtype=torch.long).to(device)

    # Hook encoder stages
    features = {}
    hooks = []
    stage_names = ['vision_encoder.stage1', 'vision_encoder.stage2',
                   'vision_encoder.stage3', 'vision_encoder.stage4']
    for name, module in model.named_modules():
        if name in stage_names:
            def make_hook(n):
                def hook_fn(m, inp, out):
                    features[n] = out.detach().cpu()
                return hook_fn
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(img_t, tokens)

    for h in hooks:
        h.remove()

    # Plot: 4 stages × 6 channels
    n_channels = 6
    fig, axes = plt.subplots(4, n_channels + 1, figsize=(18, 10),
                              gridspec_kw={'width_ratios': [2] + [1]*n_channels})
    fig.suptitle(f'{version_label} — Vision Encoder Feature Maps', fontsize=14, fontweight='bold')

    labels = ['Stage 1\n(64ch, ↓2)', 'Stage 2\n(128ch, ↓4)', 'Stage 3\n(256ch, ↓8)', 'Stage 4\n(512ch, ↓16)']

    for i, sname in enumerate(stage_names):
        feat = features[sname][0]  # (C, H, W)
        # Show input image (scaled) in first column
        axes[i, 0].imshow(denormalize(img_t[0]))
        axes[i, 0].set_title(labels[i], fontsize=9, fontweight='bold')
        axes[i, 0].axis('off')

        # Show top-activation channels
        channel_means = feat.mean(dim=(1, 2))
        top_channels = channel_means.argsort(descending=True)[:n_channels]
        for j, ch in enumerate(top_channels):
            axes[i, j+1].imshow(feat[ch].numpy(), cmap='inferno')
            axes[i, j+1].set_title(f'Ch {ch.item()}', fontsize=7)
            axes[i, j+1].axis('off')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {save_path}')


# ─── Figure 4: Text Conditioning (FiLM) ──────────────────────────────────────

def generate_text_conditioning(model, device, save_path, version_label):
    """Show how FiLM gamma/beta differ for crack vs taping prompts."""
    crack_prompt = "segment crack"
    taping_prompt = "segment taping area"

    crack_tokens = torch.tensor([tokenize(crack_prompt)], dtype=torch.long).to(device)
    taping_tokens = torch.tensor([tokenize(taping_prompt)], dtype=torch.long).to(device)

    with torch.no_grad():
        crack_film = model.text_encoder(crack_tokens)
        taping_film = model.text_encoder(taping_tokens)

    # Each element is (gamma, beta), one per FiLM stage
    stage_labels = ['Enc1 (64ch)', 'Enc2 (128ch)', 'Enc3 (256ch)', 'Enc4 (512ch)',
                    'Dec1 (256ch)', 'Dec2 (128ch)', 'Dec3 (64ch)']

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(f'{version_label} — Text Conditioning (FiLM γ/β)', fontsize=14, fontweight='bold')

    # Show first 4 encoder stages
    for i in range(min(4, len(crack_film))):
        c_gamma, c_beta = crack_film[i]
        t_gamma, t_beta = taping_film[i]

        x = np.arange(min(32, c_gamma.shape[1]))  # Show first 32 channels
        c_g = c_gamma[0, :32].cpu().numpy()
        t_g = t_gamma[0, :32].cpu().numpy()

        axes[0, i].bar(x - 0.15, c_g, 0.3, label='Crack γ', color='#D32F2F', alpha=0.7)
        axes[0, i].bar(x + 0.15, t_g, 0.3, label='Taping γ', color='#1976D2', alpha=0.7)
        axes[0, i].set_title(f'{stage_labels[i]} — Gamma (scale)', fontsize=9)
        axes[0, i].set_xlabel('Channel')
        axes[0, i].legend(fontsize=7)
        axes[0, i].grid(True, alpha=0.2)

        c_b = c_beta[0, :32].cpu().numpy()
        t_b = t_beta[0, :32].cpu().numpy()

        axes[1, i].bar(x - 0.15, c_b, 0.3, label='Crack β', color='#D32F2F', alpha=0.7)
        axes[1, i].bar(x + 0.15, t_b, 0.3, label='Taping β', color='#1976D2', alpha=0.7)
        axes[1, i].set_title(f'{stage_labels[i]} — Beta (shift)', fontsize=9)
        axes[1, i].set_xlabel('Channel')
        axes[1, i].legend(fontsize=7)
        axes[1, i].grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {save_path}')


# ─── Figure 5: Prompt Validation ─────────────────────────────────────────────

def generate_prompt_validation(model, data_root, datasets_config, image_size, device, save_path, version_label):
    """Show the SAME images with BOTH prompts to prove text conditioning works.
    Pick 2 crack images and 2 taping images, run both prompts on each → 4 rows × 4 cols."""
    transform = get_val_transforms(image_size)

    # Collect samples from both classes
    all_samples = {}
    for ds_cfg in datasets_config:
        ds_dir = os.path.join(data_root, ds_cfg['dir'], 'valid')
        mask_dir = os.path.join(ds_dir, 'masks')
        pc = ds_cfg['prompt_class']
        imgs = sorted([f for f in os.listdir(ds_dir) if f.lower().endswith(('.jpg', '.png')) and f != 'masks'])
        all_samples[pc] = []
        for img_file in imgs[:100]:
            stem = os.path.splitext(img_file)[0]
            mask_path = os.path.join(mask_dir, f'{stem}.png')
            if os.path.exists(mask_path):
                all_samples[pc].append((os.path.join(ds_dir, img_file), mask_path))

    np.random.seed(123)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle(f'{version_label} — Prompt Validation\n(Same image, different prompts → different masks)',
                 fontsize=14, fontweight='bold')

    columns = ['Input Image', 'Ground Truth', '"segment crack"', '"segment taping area"']
    for j, col in enumerate(columns):
        axes[0, j].set_title(col, fontsize=11, fontweight='bold')

    crack_tokens = torch.tensor([tokenize("segment crack")], dtype=torch.long).to(device)
    taping_tokens = torch.tensor([tokenize("segment taping area")], dtype=torch.long).to(device)

    row = 0
    for cls_name in ['crack', 'taping']:
        idxs = np.random.choice(len(all_samples[cls_name]), size=2, replace=False)
        for idx in idxs:
            img_path, mask_path = all_samples[cls_name][idx]
            img_pil = Image.open(img_path).convert('RGB')
            gt_pil = Image.open(mask_path).convert('L')

            img_np = np.array(img_pil.resize((image_size, image_size)))
            gt_np = np.array(gt_pil.resize((image_size, image_size), Image.NEAREST))
            img_t, _ = transform(img_np, gt_np)
            img_t = img_t.unsqueeze(0).to(device)
            gt_display = gt_np / 255.0 if gt_np.max() > 1 else gt_np.astype(float)

            with torch.no_grad():
                crack_mask, _ = model.predict(img_t, crack_tokens)
                taping_mask, _ = model.predict(img_t, taping_tokens)

            img_display = denormalize(img_t[0])

            axes[row, 0].imshow(img_display)
            gt_label = f'GT: {cls_name}'
            axes[row, 0].set_ylabel(f'{cls_name.upper()} image', fontsize=10, fontweight='bold')
            axes[row, 0].axis('off')

            axes[row, 1].imshow(gt_display, cmap='gray', vmin=0, vmax=1)
            axes[row, 1].axis('off')

            axes[row, 2].imshow(crack_mask[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[row, 2].axis('off')

            axes[row, 3].imshow(taping_mask[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[row, 3].axis('off')

            row += 1

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {save_path}')


# ─── Figure 6: Metrics Summary ───────────────────────────────────────────────

def generate_metrics_summary(metrics, save_path, version_label):
    """Per-class bar chart of Dice and IoU."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'{version_label} — Evaluation Metrics', fontsize=14, fontweight='bold')

    classes = ['Taping', 'Crack', 'Overall']
    dice_vals = [metrics['dice_taping'], metrics['dice_crack'], metrics['dice_all']]
    iou_vals = [metrics['iou_taping'], metrics['iou_crack'], metrics['iou_all']]

    colors = ['#388E3C', '#7B1FA2', '#1976D2']
    x = np.arange(len(classes))

    bars1 = axes[0].bar(x, dice_vals, 0.5, color=colors, alpha=0.85)
    axes[0].set_title('Dice Score', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars1, dice_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

    bars2 = axes[1].bar(x, iou_vals, 0.5, color=colors, alpha=0.85)
    axes[1].set_title('IoU Score', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars2, iou_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

    # Add mIoU annotation
    axes[1].axhline(y=metrics['miou'], color='red', linestyle='--', alpha=0.5, label=f"mIoU = {metrics['miou']:.3f}")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {save_path}')


# ─── Training History Parsers ─────────────────────────────────────────────────

def parse_v1_log(log_path):
    """Parse V1 training log into history dict."""
    history = {k: [] for k in ['train_loss', 'val_loss', 'train_dice', 'val_dice',
                                'val_dice_taping', 'val_dice_crack', 'val_miou',
                                'val_iou_taping', 'val_iou_crack']}
    with open(log_path) as f:
        for line in f:
            m = re.search(
                r'Epoch \d+ .* train_dice: ([\d.]+) .* train_loss: ([\d.]+) '
                r'.* val_dice: ([\d.]+) .* val_dice_crack: ([\d.]+) .* val_dice_taping: ([\d.]+) '
                r'.* val_iou_crack: ([\d.]+) .* val_iou_taping: ([\d.]+) '
                r'.* val_loss: ([\d.]+) .* val_miou: ([\d.]+)', line)
            if m:
                history['train_dice'].append(float(m.group(1)))
                history['train_loss'].append(float(m.group(2)))
                history['val_dice'].append(float(m.group(3)))
                history['val_dice_crack'].append(float(m.group(4)))
                history['val_dice_taping'].append(float(m.group(5)))
                history['val_iou_crack'].append(float(m.group(6)))
                history['val_iou_taping'].append(float(m.group(7)))
                history['val_loss'].append(float(m.group(8)))
                history['val_miou'].append(float(m.group(9)))
    return history


def parse_v2_history(json_path):
    """Parse V2 training_history.json."""
    with open(json_path) as f:
        data = json.load(f)
    return data


# ─── Main ─────────────────────────────────────────────────────────────────────

def make_v1_config():
    """Config for V1 model (no ASPP, no deep supervision, 256×256)."""
    with open('config/default.yaml') as f:
        cfg = yaml.safe_load(f)
    # Override for V1
    cfg['data']['image_size'] = 256
    cfg['model']['use_aspp'] = False
    cfg['model']['deep_supervision'] = False
    return cfg


def make_v2_config():
    """Config for V2 model (ASPP, deep supervision, 512×512)."""
    with open('config/default.yaml') as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices=['v1', 'v2', 'both'], default='both')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    data_root = 'data'

    versions = []
    if args.version in ('v1', 'both'):
        versions.append('v1')
    if args.version in ('v2', 'both'):
        versions.append('v2')

    for version in versions:
        print(f'\n{"="*60}')
        print(f'Generating figures for {version.upper()}')
        print(f'{"="*60}')

        if version == 'v1':
            cfg = make_v1_config()
            checkpoint = 'checkpoints_v1/v1_best.pt'
            history = parse_v1_log('runs/train_output.log')
            metrics_path = 'report/v1_evaluation_metrics.json'
            label = 'V1 (Baseline)'
        else:
            cfg = make_v2_config()
            checkpoint = 'checkpoints/best_model.pt'
            history = parse_v2_history('runs/training_history.json')
            metrics_path = 'report/evaluation_metrics.json'
            label = 'V2 (Improved)'

        image_size = cfg['data']['image_size']
        datasets_config = [
            {"dir": "drywall_taping", "prompt_class": "taping"},
            {"dir": "cracks", "prompt_class": "crack"},
        ]
        out_dir = f'report/figures/{version}'
        os.makedirs(out_dir, exist_ok=True)

        with open(metrics_path) as f:
            metrics = json.load(f)

        # Load model
        print(f'Loading model from {checkpoint}...')
        model = load_model(cfg, checkpoint, device)

        # Generate all 6 figures
        print('Generating figures...')
        generate_training_curves(history, os.path.join(out_dir, '01_training_curves.png'), label)
        generate_predictions(model, data_root, datasets_config, image_size, device,
                           os.path.join(out_dir, '02_predictions.png'), label)
        generate_encoder_features(model, data_root, datasets_config, image_size, device,
                                 os.path.join(out_dir, '03_encoder_features.png'), label)
        generate_text_conditioning(model, device,
                                  os.path.join(out_dir, '04_text_conditioning.png'), label)
        generate_prompt_validation(model, data_root, datasets_config, image_size, device,
                                  os.path.join(out_dir, '05_prompt_validation.png'), label)
        generate_metrics_summary(metrics, os.path.join(out_dir, '06_metrics_summary.png'), label)

        print(f'All figures saved to {out_dir}/')

    print('\nDone!')


if __name__ == '__main__':
    main()
