# PromptSeg-Lite: Prompted Segmentation for Drywall QA

A text-conditioned binary segmentation model I trained **entirely from scratch** (no pretrained weights). Give it a drywall image and a text prompt, and it outputs a binary mask highlighting the defect.

## What It Does

**Input**: An image + text prompt (e.g., "segment crack", "segment taping area")
**Output**: A binary mask (PNG, {0, 255}) highlighting the queried region

I built two versions — V1 as a baseline and V2 with architectural improvements:

## Results

### V1 → V2 Comparison

| Metric | V1 | V2 | Change |
|---|---|---|---|
| **mIoU** | 0.7019 | 0.7300 | **+2.81 pp** |
| Overall Dice | 0.7662 | 0.7796 | +1.34 pp |
| Taping Dice | 0.8800 | 0.9050 | +2.50 pp |
| Taping IoU | 0.8040 | 0.8472 | +4.32 pp |
| Crack Dice | 0.7377 | 0.7481 | +1.04 pp |
| Crack IoU | 0.5999 | 0.6128 | +1.29 pp |
| Parameters | 2.08M | 2.75M | +32% |
| Model Size | 8.3 MB | 11.01 MB | +33% |
| Inference | 7.76 ms | 9.70 ms | +25% |

Both versions meet the deployment constraints: model < 15 MB, inference < 25 ms.

## Architecture

### V1 — Baseline

```
Text Prompt ──→ BiLSTM ──→ FiLM (γ, β) ──┐
                                           ↓
Image ──→ MobileBlock Encoder ──→ FiLM ──→ U-Net Decoder ──→ mask
            (DSConv + SE)            ↑                         
                              skip connections               
```

- **Text Encoder**: Character-level BiLSTM → FiLM conditioning vectors
- **Vision Encoder**: Stem + 4 MobileBlock stages (DSConv + SE + FiLM)
- **Decoder**: U-Net with FiLM-conditioned skip connections
- **Loss**: DiceBCE compound loss
- **Resolution**: 256×256

### V2 — Improved

Everything from V1, plus:

- **ASPP Bottleneck**: Atrous Spatial Pyramid Pooling (rates 6, 12, 18) between encoder and decoder for multi-scale context
- **Better Loss**: Focal Tversky (α=0.7) + Boundary Loss + OHEM (top 25% hardest pixels)
- **Deep Supervision**: Auxiliary heads at intermediate decoder stages for better gradient flow
- **512×512 Resolution**: 4× more pixels for finer crack/taping detail

### Key Design Decisions

| Choice | Why I Made It |
|---|---|
| No pretrained weights | Assignment requirement — all Kaiming/Xavier/Orthogonal init |
| FiLM (not cross-attention) | Zero spatial compute, perfect for small prompt vocabulary (~10 prompts) |
| Depthwise Separable Convs | 8-9× fewer FLOPs, prevents overfitting on ~4.6K images |
| Squeeze-and-Excitation | Lightweight channel attention to focus on defect features |
| U-Net skip connections | Recovers fine spatial detail (thin cracks, narrow tape lines) |
| ASPP (V2) | Multi-scale context capture without adding many parameters |
| Focal Tversky (V2) | Focuses on false negatives — catching missed crack pixels |
| Character-level tokenization | Small vocab doesn't need word-level approaches |

## Setup

```bash
cd PromptSeg-Lite
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, PyTorch 2.0+ with CUDA, NVIDIA GPU with 8+ GB VRAM

## Data

Two Roboflow datasets:

| Dataset | Source | Prompt Class | Train | Valid |
|---|---|---|---|---|
| Drywall Taping | objectdetect-pu6rn/drywall-join-detect | "segment taping area" | 820 | 202 |
| Surface Cracks | hemant-ramphul-wfioe/surface-crack-segmentation | "segment crack" | 2,818 | 806 |

**Total**: 3,638 training / 1,008 validation samples

```bash
# Download datasets
python scripts/download_datasets.py --api-key YOUR_ROBOFLOW_API_KEY --data-root data
python scripts/prepare_masks.py --data-root data
```

## Training

```bash
# V2 training (200 epochs, 512×512, compound_v2 loss)
python train.py

# Training config is in config/default.yaml
```

### V1 Config Highlights
- Adam (lr=1e-3), batch 32, 256×256, DiceBCE loss, 150 epochs

### V2 Config Highlights
- Adam (lr=5e-4), batch 16, 512×512, Focal Tversky + Boundary + OHEM, 200 epochs
- ASPP enabled, Deep Supervision enabled, warmup 10 epochs, patience 35

## Evaluation

```bash
# Evaluate and generate predictions
python evaluate.py --config config/default.yaml --output-dir report/predictions

# Single-image inference
python predict.py path/to/image.jpg --prompt "segment crack"
python predict.py path/to/image.jpg --prompt "segment taping area" --overlay
```

## Demo App

```bash
# Launch the Streamlit demo for real-time testing
streamlit run app.py
```

The demo lets you:
- Upload any drywall image or pick from the dataset
- Choose a text prompt (crack, taping, or custom)
- Switch between V1 and V2 models
- See the segmentation mask, overlay, and probability heatmap
- Compare both prompts on the same image (prompt validation)

### Output Format
```
{image_id}__segment_crack.png
{image_id}__segment_taping_area.png
```
Single-channel PNG, values {0, 255}.

## Project Structure

```
CV_Assignment/
├── config/default.yaml            # All hyperparameters (currently V2)
├── src/
│   ├── data/
│   │   ├── dataset.py             # PromptSegDataset + DataLoader factory
│   │   ├── transforms.py          # Augmentation pipeline  
│   │   └── prompt_pool.py         # Prompt text pools + tokenizer
│   ├── model/
│   │   ├── promptseg.py           # Full model assembly
│   │   ├── text_encoder.py        # BiLSTM + FiLM generators
│   │   ├── vision_encoder.py      # MobileBlock encoder
│   │   ├── decoder.py             # U-Net decoder (+ deep supervision)
│   │   ├── aspp.py                # [V2] ASPP module
│   │   ├── film.py                # FiLM conditioning layer
│   │   └── se.py                  # Squeeze-and-Excitation block
│   ├── losses/
│   │   ├── dice_bce.py            # V1 DiceBCE compound loss
│   │   └── compound_v2.py         # [V2] Focal Tversky + Boundary + OHEM
│   ├── metrics/segmentation.py    # mIoU, Dice, per-class tracking
│   └── utils/
│       ├── hooks.py               # Feature/gradient capture
│       ├── logger.py              # TensorBoard + file logging
│       └── visualization.py       # Feature maps, predictions, curves
├── train.py                       # Training script with AMP
├── evaluate.py                    # Evaluation + mask generation
├── predict.py                     # Single-image inference
├── app.py                         # Streamlit demo app
├── generate_figures.py            # Uniform figure generation (V1 & V2)
├── monitor.py                     # Training monitor
├── generate_course_materials.py   # DL course slide materials
├── scripts/
│   ├── download_datasets.py       # Roboflow data download
│   └── prepare_masks.py           # COCO → binary mask
├── report/
│   ├── report.md                  # Technical report (V1 + V2)
│   ├── evaluation_metrics.json    # V2 metrics
│   ├── v1_evaluation_metrics.json # V1 metrics
│   ├── figures/
│   │   ├── v1/                    # V1 figures (6 uniform figures)
│   │   │   ├── 01_training_curves.png
│   │   │   ├── 02_predictions.png
│   │   │   ├── 03_encoder_features.png
│   │   │   ├── 04_text_conditioning.png
│   │   │   ├── 05_prompt_validation.png
│   │   │   └── 06_metrics_summary.png
│   │   └── v2/                    # V2 figures (same 6 figures for comparison)
│   │       ├── 01_training_curves.png
│   │       ├── 02_predictions.png
│   │       ├── 03_encoder_features.png
│   │       ├── 04_text_conditioning.png
│   │       ├── 05_prompt_validation.png
│   │       └── 06_metrics_summary.png
│   ├── predictions/               # V2 prediction masks (1,008)
│   └── predictions_v1/            # V1 prediction masks (1,008)
├── checkpoints/                   # V2 model checkpoints
└── checkpoints_v1/                # V1 best checkpoint backup
```

## Reproducibility

- **Seed**: 42 (Python, NumPy, PyTorch, CUDA)
- **Deterministic**: `torch.backends.cudnn.deterministic = True`
- **Hardware**: NVIDIA L4 GPU (24GB), CUDA 12.2, PyTorch 2.5.1+cu121, Python 3.11.2

## License

Datasets are licensed under CC BY 4.0.
