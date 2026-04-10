# PromptSeg-Lite: Technical Report

## 1. Introduction

I built **PromptSeg-Lite**, a text-conditioned binary segmentation model for drywall quality assurance. The idea is simple: give the model a drywall image and a text prompt like "segment crack" or "segment taping area", and it outputs a binary mask highlighting that defect. The whole model is trained from scratch — no pretrained weights at all.

I developed two versions:
- **V1**: My baseline model to get something working end-to-end
- **V2**: An improved version where I addressed V1's weaknesses with better architecture and loss design

This report covers both versions so you can see the progression and what each change actually did.

## 2. Approach

### 2.1 Base Architecture (V1)

I designed PromptSeg-Lite with three components:

1. **Text Encoder**: A character-level BiLSTM that converts text prompts into conditioning vectors. I went with character-level tokenization because my prompt vocabulary is tiny (about 10 prompts, ~5 words each). The BiLSTM output gets pooled and projected to generate FiLM parameters (γ, β) for each encoder and decoder stage.

2. **Vision Encoder**: A MobileBlock-based encoder with 5 stages. Each stage uses Depthwise Separable Convolutions (8-9× fewer FLOPs than standard convs), Squeeze-and-Excitation blocks for channel attention, and FiLM conditioning so the text prompt can modulate the visual features.

3. **U-Net Decoder**: 3 stages with skip connections, followed by an upsample stage and a 1×1 conv head. FiLM conditioning is applied at each decoder stage too.

### 2.2 V2 Architectural Upgrades

After analyzing V1's results, I noticed the model was struggling with fine crack boundaries and wasn't capturing enough multi-scale context. So I made five key changes:

1. **ASPP Module (Atrous Spatial Pyramid Pooling)**: I added an ASPP block between the encoder and decoder. It uses dilated convolutions at rates 6, 12, 18 plus a global average pooling branch — this lets the model capture context at multiple scales before decoding. I used depthwise separable convolutions inside ASPP to keep the parameter count reasonable.

2. **Focal Tversky + Boundary + OHEM Loss**: V1 used a simple DiceBCE loss. For V2, I switched to a compound loss with three parts:
   - Focal Tversky Loss (α=0.7) to focus on false negatives (missed crack pixels)
   - Boundary Loss to sharpen mask edges
   - Online Hard Example Mining (OHEM, top 25%) to focus training on the hardest pixels

3. **Deep Supervision**: I added auxiliary prediction heads at intermediate decoder stages. During training, these extra heads provide gradient signal deeper in the network, which helps the model learn better features.

4. **512×512 Resolution**: I increased input resolution from 256×256 to 512×512. This preserves more fine detail, especially for thin cracks.

5. **Longer Training**: I trained for 200 epochs (vs. 150 in V1) with a higher patience of 35 epochs and a 10-epoch warmup.

### 2.3 Key Design Choices

- **FiLM over Cross-Attention**: FiLM adds zero spatial computation — just channel-wise affine transforms. With only ~10 prompt variants, cross-attention would be over-parameterized.
- **Depthwise Separable Convolutions**: Critical for preventing overfitting on ~4.6K training images.
- **Character-level tokenization**: No need for a word vocabulary; naturally handles prompt variants.

### 2.4 Initialization

All weights initialized from scratch:
- Convolutional layers: Kaiming Normal
- Linear layers: Xavier Uniform
- LSTM: Orthogonal
- Batch normalization: weight=1, bias=0
- FiLM generators: Identity-like (γ≈1, β≈0) to start with unmodified features

## 3. Data

### 3.1 Datasets

| Dataset | Source | Annotation Type | Train | Valid |
|---|---|---|---|---|
| Drywall Taping | Roboflow: drywall-join-detect v2 | COCO bbox → filled masks | 820 | 202 |
| Surface Cracks | Roboflow: surface-crack-segmentation | COCO polygon annotations | 2,818 | 806 |

**Total**: 3,638 training samples, 1,008 validation samples

### 3.2 Data Splits

- **Drywall taping**: I used the original Roboflow train/valid splits. Bbox annotations are converted to filled rectangular binary masks — this works fine since drywall joints are roughly rectangular.
- **Surface cracks**: This dataset has detailed polygon annotations (average 25+ vertices per annotation), giving high-quality crack boundary masks.

### 3.3 Augmentation

I designed the augmentations for construction-site conditions:
- Random horizontal flip (p=0.5), vertical flip (p=0.3)
- Random rotation (±15°)
- Color jitter (brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1) — simulates harsh site lighting
- Gaussian blur (p=0.3) — simulates motion blur from a moving robot
- Gaussian noise (p=0.2) — simulates dusty/dirty camera lens

Validation uses only deterministic resize + normalize.

## 4. Training Configuration

### V1 Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Scheduler | Cosine annealing with 5-epoch linear warmup |
| Loss | DiceLoss + BCEWithLogits (equal weighting) |
| Batch size | 32 |
| Image size | 256×256 |
| Gradient clipping | Max norm 5.0 |
| AMP | Enabled |
| Early stopping | Patience 25 epochs |
| Epochs | 150 |

### V2 Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=5e-4, weight_decay=1e-4) |
| Scheduler | Cosine annealing with 10-epoch linear warmup |
| Loss | Focal Tversky (1.0) + Boundary (0.5) + OHEM (0.5) |
| Batch size | 16 |
| Image size | 512×512 |
| Gradient clipping | Max norm 5.0 |
| AMP | Enabled |
| Early stopping | Patience 35 epochs |
| Epochs | 200 |
| Deep supervision | Enabled (aux weights: 0.4, 0.2) |
| ASPP | Enabled (rates: 6, 12, 18) |

I reduced the batch size from 32 to 16 because of the larger image size (512×512 uses 4× more GPU memory per image). I also lowered the learning rate from 1e-3 to 5e-4 since the new loss function has more components and I wanted more stable convergence.

## 5. Results

### 5.1 V1 Results (Baseline)

Training completed all 150 epochs in ~40 minutes on an NVIDIA L4 GPU (24GB). Best model was at epoch 146.

**V1 Metrics:**

| Metric | Overall | Taping | Crack |
|---|---|---|---|
| Dice | 0.7662 | 0.8800 | 0.7377 |
| IoU | 0.6407 | 0.8040 | 0.5999 |
| mIoU | 0.7019 | — | — |

**V1 Speed:**

| Metric | Value |
|---|---|
| Parameters | 2,077,969 (~2.08M) |
| Model size (FP32) | 8.3 MB |
| Inference latency | 7.76 ms/image |
| Throughput | 129 FPS |

V1 gave me a solid baseline — taping detection was strong (Dice 0.88) but crack IoU was only 0.60, which I wanted to improve.

**V1 Figures** (all in `figures/v1/`):
- `01_training_curves.png` — Loss, Dice, and mIoU convergence over 150 epochs
- `02_predictions.png` — Image | GT | Prediction examples for both classes
- `03_encoder_features.png` — Vision encoder feature maps at stages 1-4
- `04_text_conditioning.png` — FiLM γ/β values showing how crack vs taping prompts modulate features differently
- `05_prompt_validation.png` — Same image with both prompts → different masks (proves text conditioning works)
- `06_metrics_summary.png` — Per-class Dice and IoU bar charts

### 5.2 V2 Results (Improved)

Training completed all 200 epochs in ~1.5 hours on the same L4 GPU. Best model was at epoch 191.

**V2 Metrics:**

| Metric | Overall | Taping | Crack |
|---|---|---|---|
| Dice | 0.7796 | 0.9050 | 0.7481 |
| IoU | 0.6598 | 0.8472 | 0.6128 |
| mIoU | 0.7300 | — | — |

**V2 Speed:**

| Metric | Value |
|---|---|
| Parameters | 2,752,177 (~2.75M) |
| Model size (FP32) | 11.01 MB |
| Inference latency | 9.70 ms/image |
| Throughput | 103 FPS |

### 5.3 V1 → V2 Improvement Summary

| Metric | V1 | V2 | Change |
|---|---|---|---|
| mIoU | 0.7019 | 0.7300 | **+2.81 pp** |
| Overall Dice | 0.7662 | 0.7796 | **+1.34 pp** |
| Taping Dice | 0.8800 | 0.9050 | **+2.50 pp** |
| Taping IoU | 0.8040 | 0.8472 | **+4.32 pp** |
| Crack Dice | 0.7377 | 0.7481 | **+1.04 pp** |
| Crack IoU | 0.5999 | 0.6128 | **+1.29 pp** |
| Parameters | 2.08M | 2.75M | +32% |
| Model Size | 8.3 MB | 11.01 MB | +33% |
| Inference | 7.76 ms | 9.70 ms | +25% |

The results show consistent improvement across all metrics. The biggest gains were in taping segmentation (+4.3 pp IoU), which makes sense because the higher resolution and ASPP module help capture the full extent of taping regions. Crack detection also improved, though more modestly — cracks are inherently harder because they're thin and irregular.

The model size grew from 8.3 MB to 11.01 MB (still well under the 15 MB constraint), and inference slowed slightly from 7.76 ms to 9.70 ms (still well under the 25 ms requirement).

**V2 Figures** (all in `figures/v2/` — same structure as V1 for direct comparison):
- `01_training_curves.png` — Loss, Dice, and mIoU convergence over 200 epochs
- `02_predictions.png` — Image | GT | Prediction examples for both classes
- `03_encoder_features.png` — Vision encoder feature maps at stages 1-4 (compare with V1 to see richer features)
- `04_text_conditioning.png` — FiLM γ/β values for crack vs taping
- `05_prompt_validation.png` — Same image, both prompts → different masks
- `06_metrics_summary.png` — Per-class Dice and IoU bar charts

## 6. DL Course Materials

### Slide 1 — Input Tensor
- **V1**: Image `(B, 3, 256, 256)` float32, ImageNet-normalized; Prompt tokens `(B, 21)` int64
- **V2**: Image `(B, 3, 512, 512)` float32, ImageNet-normalized; Prompt tokens `(B, 21)` int64

### Slide 2 — Major Blocks
1. Text Encoder: Embedding → BiLSTM → FC → FiLM generators
2. Vision Encoder: Stem + 4 MobileBlock stages (DSConv + SE + FiLM)
3. **[V2]** ASPP Bottleneck: 1×1 + dilated 3×3 (rates 6, 12, 18) + global pool → fuse
4. U-Net Decoder: 3 skip stages + upsample + 1×1 head
5. **[V2]** Deep Supervision: auxiliary heads at decoder stages 1 and 2

### Slides 3-5 — Layer Specifications
See `report/layer_specs.json` — detailed layer-by-layer shapes, kernel sizes, and parameter counts.

### Slide 6 — Feature Maps & Text Conditioning
- Vision encoder features: `figures/v1/03_encoder_features.png` and `figures/v2/03_encoder_features.png`
- Text conditioning (FiLM): `figures/v1/04_text_conditioning.png` and `figures/v2/04_text_conditioning.png`
- Prompt validation: `figures/v1/05_prompt_validation.png` and `figures/v2/05_prompt_validation.png`

### Slide 7 — Loss & Optimizer
- **V1 Loss**: DiceBCE compound (Dice handles imbalance, BCE provides stable gradients)
- **V2 Loss**: Focal Tversky (α=0.7, focuses on false negatives) + Boundary Loss (sharpens edges) + OHEM (mines hardest 25% of pixels)
- Optimizer: Adam with cosine annealing + linear warmup
- AMP training for 2× throughput

### Slide 8 — Training Curves
- V1 curves: `figures/v1/01_training_curves.png` (150 epochs)
- V2 curves: `figures/v2/01_training_curves.png` (200 epochs)

### Slide 9 — Receptive Field & Complexity
**V1:**
- Receptive field at bottleneck: 63×63 pixels (24.6% coverage of 256×256)
- Parameters: 2.08M | Model size: 8.3 MB | Inference: 7.76 ms

**V2:**
- ASPP expands effective receptive field with dilated convolutions (rates 6, 12, 18)
- Parameters: 2.75M | Model size: 11.01 MB | Inference: 9.70 ms

## 7. Analysis and Failure Cases

### What Worked Well
- **FiLM conditioning** does a great job distinguishing between crack and taping prompts — the model clearly activates different features for each.
- **ASPP in V2** helped capture larger taping regions more completely.
- **Focal Tversky loss** reduced false negatives on thin cracks compared to V1's DiceBCE.

### What I'd Improve Next
1. **Very thin cracks**: Some hairline cracks (1-2 pixels wide) still get missed, even at 512×512 resolution.
2. **Ambiguous taping regions**: Fresh drywall mud that blends with the wall surface is hard to detect.
3. **Crack IoU still below 0.65**: Cracks are inherently difficult — they're thin, irregular, and vary a lot in contrast. I think adding attention mechanisms or a dedicated boundary refinement module could help here.
4. **More data would help**: With only 820 taping images, the model is somewhat data-limited for that class.

## 8. Hardware and Reproducibility

| Detail | Value |
|---|---|
| GPU | NVIDIA L4 (24GB VRAM) |
| CUDA | 12.2 |
| PyTorch | 2.5.1+cu121 |
| Python | 3.11.2 |
| Seed | 42 (Python, NumPy, PyTorch, CUDA) |
| Deterministic | `torch.backends.cudnn.deterministic = True` |
| V1 training time | ~40 minutes |
| V2 training time | ~1.5 hours |
| Peak VRAM (V2) | 3.81 GB (15.9% of 24 GB) |

## 9. Conclusion

I built PromptSeg-Lite from scratch and showed that text-conditioned segmentation works even with a small dataset (~4.6K images) and lightweight architecture. The V1 baseline gave solid results (mIoU 0.70), and V2's architectural improvements — ASPP, better loss function, deep supervision, and higher resolution — pushed that to mIoU 0.73 with consistent gains across both classes.

The model stays small (11 MB) and fast (9.7 ms inference), meeting Origin's edge deployment constraints. The FiLM conditioning mechanism lets a single model handle both defect types without needing separate models or pretrained encoders, which is what makes this approach practical for real-world construction QA.
