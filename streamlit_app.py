"""
Lung Cancer Detection — Dual-Stream Pseudo-3D VGG + Handcrafted Features
Streamlit Cloud demo app for the model trained in
`lung-cancer-detection-using-deep-learning-techniqu.ipynb`.

This app re-implements, EXACTLY, the preprocessing, architecture, and
Grad-CAM logic from the notebook so that inference here matches training.
"""

import os
import io
import tempfile

import cv2
import requests
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from skimage.feature import hog, local_binary_pattern
from PIL import Image
import matplotlib.pyplot as plt

# =====================================================================
# CONFIG
# =====================================================================
MODEL_URL = "https://github.com/blurerjr/lung-cancer-app/releases/download/model/dual_stream_model.pth"
TARGET_SHAPE = (224, 224)
CLASS_NAMES = ["Normal", "Benign", "Malignant"]
CLASS_COLORS = {"Normal": "#2e7d32", "Benign": "#f9a825", "Malignant": "#c62828"}

st.set_page_config(
    page_title="Lung Cancer Detection — Dual-Stream Model",
    page_icon="🫁",
    layout="wide",
)

# =====================================================================
# 1. HANDCRAFTED FEATURE EXTRACTION ENGINE  (identical to notebook)
# =====================================================================
def extract_handcrafted_features(image_np):
    """Extracts HOG and LBP descriptors from a 2D grayscale image slice."""
    if image_np.dtype != np.uint8:
        img_uint8 = (image_np * 255).astype(np.uint8)
    else:
        img_uint8 = image_np

    if img_uint8.shape != TARGET_SHAPE:
        img_uint8 = cv2.resize(img_uint8, TARGET_SHAPE, interpolation=cv2.INTER_AREA)

    hog_feats = hog(
        img_uint8, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), visualize=False,
    )

    lbp = local_binary_pattern(img_uint8, P=24, R=3, method="uniform")
    # Uniform LBP with P=24 has a fixed P+2=26 categories (P uniform patterns
    # + 2 edge cases). We bin on this fixed range rather than lbp.max()+1 so
    # every image produces the same feature length, regardless of its content
    # (a content-dependent bin count risks a dimension mismatch against the
    # trained model on images with less texture variety).
    n_bins = 24 + 2
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    combined_handcrafted = np.concatenate([hog_feats, lbp_hist])
    return torch.tensor(combined_handcrafted, dtype=torch.float32)


# =====================================================================
# 2. PREPROCESSING  (identical to DualStreamDataset._preprocess_slice)
# =====================================================================
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_slice(pil_image):
    """PIL image -> CLAHE-enhanced, resized, normalized [0,1] float32 grayscale array."""
    img = np.array(pil_image.convert("L"))  # grayscale
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    img_enhanced = _clahe.apply(img_blur)
    img_resized = cv2.resize(img_enhanced, TARGET_SHAPE, interpolation=cv2.INTER_AREA)
    return img_resized.astype(np.float32) / 255.0


def build_pseudo3d_tensor(prev_img, curr_img, next_img):
    """Three PIL images (prev, curr, next) -> normalized (1,3,224,224) tensor + center slice [0,1]."""
    s_prev = preprocess_slice(prev_img)
    s_curr = preprocess_slice(curr_img)
    s_next = preprocess_slice(next_img)

    pseudo_3d = np.stack([s_prev, s_curr, s_next], axis=0)
    tensor = torch.tensor(pseudo_3d, dtype=torch.float32)
    tensor = (tensor - 0.5) / 0.5  # medical-image normalization used in training
    return tensor.unsqueeze(0), s_curr


# =====================================================================
# 3. MODEL ARCHITECTURE  (identical to DualStreamPseudo3DVGG)
# =====================================================================
class DualStreamPseudo3DVGG(nn.Module):
    def __init__(self, handcrafted_dim, num_classes=3):
        super().__init__()
        vgg = models.vgg16(weights=None)  # weights loaded from checkpoint, not ImageNet
        self.vgg_features = vgg.features
        self.vgg_pool = vgg.avgpool

        self.vgg_fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        self.handcrafted_projector = nn.Sequential(
            nn.Linear(handcrafted_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )

        self.fusion_classifier = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_img, x_handcrafted):
        x1 = self.vgg_features(x_img)
        x1 = self.vgg_pool(x1)
        x1 = torch.flatten(x1, 1)
        vgg_vector = self.vgg_fc(x1)

        hc_vector = self.handcrafted_projector(x_handcrafted)

        fused = torch.cat((vgg_vector, hc_vector), dim=1)
        return self.fusion_classifier(fused)


# =====================================================================
# 4. GRAD-CAM  (identical to notebook's DualStreamGradCAM)
# =====================================================================
class DualStreamGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        self._hook_layers()

    def _hook_layers(self):
        def forward_hook(module, inp, output):
            self.features = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, x_img, x_handcrafted, target_class):
        self.model.eval()
        output = self.model(x_img, x_handcrafted)

        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        features = self.features.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(features.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * features[i, :, :]

        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        cam = cv2.resize(cam, TARGET_SHAPE)
        return cam, output


def overlay_heatmap(center_slice_01, heatmap, alpha=0.4):
    """center_slice_01: float32 [0,1] grayscale array. Returns RGB overlay uint8 image."""
    img_gray = (center_slice_01 * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlayed = cv2.addWeighted(heatmap_color, alpha, img_rgb, 1 - alpha, 0)
    return img_rgb, heatmap_color, overlayed


# =====================================================================
# 5. MODEL LOADING (downloads the .pth from the GitHub release, cached)
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    # Download weights to a temp file (cached for the life of the app process)
    weights_path = os.path.join(tempfile.gettempdir(), "dual_stream_model.pth")
    if not os.path.exists(weights_path):
        with requests.get(MODEL_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(weights_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    state_dict = torch.load(weights_path, map_location="cpu")

    # Infer the handcrafted feature dimension straight from the checkpoint
    # (the first Linear layer of handcrafted_projector) rather than
    # recomputing it locally — this is robust to any drift between this
    # app's HOG/LBP params and whatever the model was actually trained with.
    handcrafted_dim = state_dict["handcrafted_projector.0.weight"].shape[1]

    model = DualStreamPseudo3DVGG(handcrafted_dim=handcrafted_dim, num_classes=len(CLASS_NAMES))
    model.load_state_dict(state_dict)
    model.eval()
    return model, handcrafted_dim


# =====================================================================
# 6. STREAMLIT UI
# =====================================================================
st.title("🫁 Lung Cancer Detection — Dual-Stream Model")
st.caption(
    "Pseudo-3D VGG16 (deep spatial stream) + HOG/LBP texture stream, fused for "
    "3-class classification: **Normal / Benign / Malignant**."
)
st.warning(
    "⚠️ Research / model-evaluation tool only. This is **not** a medical device "
    "and must not be used for real diagnostic or clinical decisions.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Model")
    st.markdown(f"**Weights source:**\n\n`{MODEL_URL}`")
    status = st.empty()
    try:
        status.info("Loading model (first run downloads the weights)…")
        model, handcrafted_dim = load_model()
        status.success(f"Model ready · handcrafted feature dim = {handcrafted_dim}")
    except Exception as e:
        status.error(f"Failed to load model: {e}")
        st.stop()

    st.divider()
    st.header("Input mode")
    mode = st.radio(
        "How do you want to provide the scan?",
        ["Single image (recommended)", "Three consecutive slices (prev / current / next)"],
        help=(
            "The model was trained on triplets of consecutive CT slices "
            "(a 'pseudo-3D' input). If you only have one image, it is "
            "duplicated across all three channels — a reasonable "
            "approximation, but the model performs best on true adjacent slices."
        ),
    )

# ---- Image input ----
prev_img = curr_img = next_img = None

if mode.startswith("Single"):
    uploaded = st.file_uploader("Upload a lung CT slice", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        curr_img = Image.open(io.BytesIO(uploaded.read()))
        prev_img = curr_img
        next_img = curr_img
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        f_prev = st.file_uploader("Previous slice", type=["png", "jpg", "jpeg"], key="prev")
    with c2:
        f_curr = st.file_uploader("Current slice (center)", type=["png", "jpg", "jpeg"], key="curr")
    with c3:
        f_next = st.file_uploader("Next slice", type=["png", "jpg", "jpeg"], key="next")
    if f_prev and f_curr and f_next:
        prev_img = Image.open(io.BytesIO(f_prev.read()))
        curr_img = Image.open(io.BytesIO(f_curr.read()))
        next_img = Image.open(io.BytesIO(f_next.read()))

run = st.button("🔬 Run inference", type="primary", disabled=curr_img is None)

if curr_img is not None and run:
    with st.spinner("Running preprocessing, inference, and Grad-CAM…"):
        img_tensor, center_slice_01 = build_pseudo3d_tensor(prev_img, curr_img, next_img)
        handcrafted_vec = extract_handcrafted_features(center_slice_01).unsqueeze(0)

        # --- Grad-CAM needs gradients enabled ---
        vgg_last_layer = model.vgg_features[-1]
        cam_engine = DualStreamGradCAM(model, vgg_last_layer)

        img_tensor.requires_grad_(False)
        with torch.set_grad_enabled(True):
            # First a plain forward pass to get probabilities without
            # holding onto a graph we don't need twice.
            with torch.no_grad():
                logits = model(img_tensor, handcrafted_vec)
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = int(torch.argmax(probs).item())
                pred_name = CLASS_NAMES[pred_idx]
                confidence = float(probs[pred_idx].item()) * 100

            heatmap, _ = cam_engine.generate_heatmap(img_tensor, handcrafted_vec, target_class=pred_idx)

        img_rgb, heatmap_color, overlay = overlay_heatmap(center_slice_01, heatmap, alpha=0.4)

    # ---- Results ----
    st.subheader("Prediction")
    badge_color = CLASS_COLORS.get(pred_name, "#333")
    st.markdown(
        f"<h2 style='color:{badge_color};'>{pred_name} — {confidence:.2f}% confidence</h2>",
        unsafe_allow_html=True,
    )

    prob_cols = st.columns(len(CLASS_NAMES))
    for i, cname in enumerate(CLASS_NAMES):
        with prob_cols[i]:
            st.metric(cname, f"{probs[i].item()*100:.2f}%")
            st.progress(float(probs[i].item()))

    st.divider()

    # ---- Visual Verification (Grad-CAM) ----
    st.subheader("Visual Verification — Grad-CAM")
    st.caption(
        "Highlights the image regions that most influenced the model's "
        f"'{pred_name}' prediction (last VGG16 convolutional block)."
    )

    vcol1, vcol2, vcol3 = st.columns(3)
    with vcol1:
        st.image(img_rgb, caption="Original (center slice)", use_container_width=True)
    with vcol2:
        st.image(heatmap_color, caption="Raw Grad-CAM heatmap", use_container_width=True)
    with vcol3:
        st.image(overlay, caption=f"Overlay — Predicted: {pred_name}", use_container_width=True)

    # Downloadable combined figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(img_rgb); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM heatmap"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title(f"Overlay — {pred_name} ({confidence:.1f}%)"); axes[2].axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    st.download_button(
        "⬇️ Download visualization",
        data=buf.getvalue(),
        file_name="gradcam_result.png",
        mime="image/png",
    )

elif curr_img is None:
    st.info("Upload an image (or a slice triplet) and click **Run inference** to see results.")
