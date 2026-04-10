"""PromptSeg-Lite: Streamlit Demo App for real-time inference."""

import os
import sys
import yaml
import torch
import numpy as np
from PIL import Image
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from src.model.promptseg import PromptSegLite
from src.data.prompt_pool import tokenize, PROMPT_POOL
from src.data.transforms import get_val_transforms


# ─── Config ──────────────────────────────────────────────────────────────────

V1_CHECKPOINT = "checkpoints_v1/v1_best.pt"
V2_CHECKPOINT = "checkpoints/best_model.pt"
CONFIG_PATH = "config/default.yaml"

PROMPT_OPTIONS = {
    "Segment Crack": "segment crack",
    "Segment Wall Crack": "segment wall crack",
    "Segment Surface Crack": "segment surface crack",
    "Segment Taping Area": "segment taping area",
    "Segment Joint Tape": "segment joint tape",
    "Segment Drywall Seam": "segment drywall seam",
}


@st.cache_resource
def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_model_cached(version):
    """Load model with caching so it's only loaded once."""
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = dict(cfg["model"])
    if version == "v1":
        model_cfg["use_aspp"] = False
        model_cfg["deep_supervision"] = False
        checkpoint_path = V1_CHECKPOINT
        image_size = 256
    else:
        checkpoint_path = V2_CHECKPOINT
        image_size = cfg["data"]["image_size"]

    model = PromptSegLite(model_cfg)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state.items() if k in model_keys}
    model.load_state_dict(filtered, strict=False)
    model = model.to(device)
    model.eval()
    return model, image_size, device


def run_inference(model, image_pil, prompt_text, image_size, device):
    """Run model inference on a PIL image."""
    transform = get_val_transforms(image_size)
    img_np = np.array(image_pil.convert("RGB").resize((image_size, image_size)))
    dummy_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    img_t, _ = transform(img_np, dummy_mask)
    img_t = img_t.unsqueeze(0).to(device)

    tokens = torch.tensor([tokenize(prompt_text)], dtype=torch.long).to(device)

    with torch.no_grad():
        mask, probs = model.predict(img_t, tokens)

    mask_np = mask[0, 0].cpu().numpy()  # (H, W) binary
    probs_np = probs[0, 0].cpu().numpy()  # (H, W) probabilities
    return mask_np, probs_np


def create_overlay(image_pil, mask_np, color=(255, 0, 0), alpha=0.4):
    """Create a red overlay of the mask on the image."""
    img = np.array(image_pil.convert("RGB").resize(
        (mask_np.shape[1], mask_np.shape[0])))
    overlay = img.copy()
    mask_bool = mask_np > 0.5
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + alpha * np.array(color)
    ).astype(np.uint8)
    return Image.fromarray(overlay)


# ─── Streamlit UI ────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="PromptSeg-Lite Demo",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 PromptSeg-Lite: Text-Prompted Segmentation")
    st.markdown(
        "Upload a drywall image and select a text prompt to generate a segmentation mask. "
        "The model was trained from scratch — no pretrained weights."
    )

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        version = st.radio(
            "Model Version",
            ["V2 (Improved)", "V1 (Baseline)"],
            index=0,
        )
        version_key = "v2" if "V2" in version else "v1"

        prompt_label = st.selectbox(
            "Text Prompt",
            list(PROMPT_OPTIONS.keys()),
            index=0,
        )
        prompt_text = PROMPT_OPTIONS[prompt_label]

        custom_prompt = st.text_input(
            "Or type a custom prompt:",
            placeholder="e.g., segment crack line",
        )
        if custom_prompt.strip():
            prompt_text = custom_prompt.strip()

        st.markdown("---")
        st.markdown(f"**Active prompt**: `{prompt_text}`")
        st.markdown(f"**Model**: {version}")

        st.markdown("---")
        st.header("Model Info")
        if version_key == "v1":
            st.markdown("- **Params**: 2.08M\n- **Size**: 8.3 MB\n- **Resolution**: 256×256")
        else:
            st.markdown("- **Params**: 2.75M\n- **Size**: 11.01 MB\n- **Resolution**: 512×512")

    # Main area
    uploaded = st.file_uploader(
        "Upload a drywall surface image",
        type=["jpg", "jpeg", "png"],
    )

    # Also allow using example images from validation set
    use_example = st.checkbox("Or use an example image from the dataset")
    example_path = None
    if use_example:
        example_dir_crack = "data/cracks/valid"
        example_dir_taping = "data/drywall_taping/valid"
        examples = []
        if os.path.exists(example_dir_crack):
            crack_imgs = sorted([
                f for f in os.listdir(example_dir_crack)
                if f.lower().endswith(('.jpg', '.png')) and f != 'masks'
            ])[:10]
            examples += [(os.path.join(example_dir_crack, f), f"crack: {f}") for f in crack_imgs]
        if os.path.exists(example_dir_taping):
            taping_imgs = sorted([
                f for f in os.listdir(example_dir_taping)
                if f.lower().endswith(('.jpg', '.png')) and f != 'masks'
            ])[:10]
            examples += [(os.path.join(example_dir_taping, f), f"taping: {f}") for f in taping_imgs]

        if examples:
            selected = st.selectbox(
                "Select example image",
                [label for _, label in examples],
            )
            example_path = [p for p, l in examples if l == selected][0]

    # Process
    image_pil = None
    if uploaded is not None:
        image_pil = Image.open(uploaded)
    elif example_path is not None:
        image_pil = Image.open(example_path)

    if image_pil is not None:
        # Load model
        with st.spinner(f"Loading {version} model..."):
            model, image_size, device = load_model_cached(version_key)

        # Run inference
        with st.spinner("Running inference..."):
            mask_np, probs_np = run_inference(
                model, image_pil, prompt_text, image_size, device
            )

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Input Image")
            st.image(image_pil, use_container_width=True)

        with col2:
            st.subheader("Predicted Mask")
            st.image(mask_np, use_container_width=True, clamp=True)

        with col3:
            st.subheader("Overlay")
            overlay = create_overlay(image_pil, mask_np)
            st.image(overlay, use_container_width=True)

        # Probability heatmap
        with st.expander("Show probability heatmap"):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            im = ax.imshow(probs_np, cmap='hot', vmin=0, vmax=1)
            ax.axis('off')
            ax.set_title(f'Probability Map — "{prompt_text}"')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            plt.close(fig)

        # Prompt comparison: run BOTH prompts on the same image
        with st.expander("Compare both prompts on this image"):
            st.markdown("Shows how the model responds to different text prompts on the same image:")
            crack_mask, _ = run_inference(model, image_pil, "segment crack", image_size, device)
            taping_mask, _ = run_inference(model, image_pil, "segment taping area", image_size, device)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(image_pil, caption="Input Image", use_container_width=True)
            with c2:
                st.image(crack_mask, caption='"segment crack"', use_container_width=True, clamp=True)
            with c3:
                st.image(taping_mask, caption='"segment taping area"', use_container_width=True, clamp=True)

        # Stats
        fg_pixels = np.sum(mask_np > 0.5)
        total_pixels = mask_np.size
        fg_pct = 100 * fg_pixels / total_pixels
        st.markdown(
            f"**Segmentation stats**: {fg_pixels:,} foreground pixels "
            f"({fg_pct:.1f}% of image) | "
            f"Mean probability: {probs_np.mean():.3f} | "
            f"Max probability: {probs_np.max():.3f}"
        )
    else:
        st.info("Upload an image or select an example to get started.")


if __name__ == "__main__":
    main()
