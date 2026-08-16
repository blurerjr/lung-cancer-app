"""
Lung Cancer Detection — Dual-Stream Pseudo-3D VGG + Handcrafted Features
Streamlit demo for the model trained in
`lung-cancer-detection-using-deep-learning-techniqu.ipynb`.

Preprocessing, architecture and Grad-CAM logic are unchanged from the
notebook so that inference here matches training. Everything else in this
file is presentation: a dark reading-room interface, a gallery of sample
studies pulled from the project repository, and a segmented probability
readout in place of three separate progress bars.
"""

import io
import os
import re
import tempfile
import urllib.parse

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

GH_OWNER = "blurerjr"
GH_REPO = "lung-cancer-app"
GH_BRANCH = "master"
GH_SAMPLE_DIR = "test_mamography"
GH_API_URL = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{GH_SAMPLE_DIR}?ref={GH_BRANCH}"
GH_RAW_BASE = f"https://raw.githubusercontent.com/{GH_OWNER}/{GH_REPO}/{GH_BRANCH}/{GH_SAMPLE_DIR}/"

# Used only if the GitHub API is unreachable or rate-limited at runtime.
FALLBACK_SAMPLES = [
    "Normal case (10)(6).jpg",
    "Normal case (103).jpg",
    "Normal case (103)(6).jpg",
    "Normal case (107)(6).jpg",
    "Benign case (101)(9).jpg",
    "Benign case (101)(14).jpg",
    "Benign case (101)(21).jpg",
    "Benign case (101)(25).jpg",
    "Malignant case (100)(4).jpg",
    "Malignant case (100)(7).jpg",
    "Malignant case (106)(6).jpg",
    "Malignant case (109)(6).jpg",
]

TARGET_SHAPE = (224, 224)
CLASS_NAMES = ["Normal", "Benign", "Malignant"]
CLASS_COLORS = {"Normal": "#4ADE80", "Benign": "#F5B942", "Malignant": "#FF5C6C"}

st.set_page_config(
    page_title="Lung Cancer Detection — Dual-Stream Model",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# STYLING
# =====================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --ink: #0B1013;
  --panel: #131B20;
  --line: #22323A;
  --text: #E3EBEE;
  --muted: #8CA0A9;
  --accent: #5FD3C4;
  --normal: #4ADE80;
  --benign: #F5B942;
  --malignant: #FF5C6C;
}

.stApp {
  background: radial-gradient(1100px 620px at 18% -12%, #17262D 0%, var(--ink) 58%);
  color: var(--text);
  font-family: 'Inter', system-ui, sans-serif;
}

#MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }

h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; color: var(--text); letter-spacing: -0.01em; }
p, li, label, .stMarkdown { color: var(--text); }

/* ---------- masthead ---------- */
.masthead { border-bottom: 1px solid var(--line); padding-bottom: 1.4rem; margin-bottom: 2.2rem; }
.masthead .eyebrow {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--accent); display: block; margin-bottom: 0.55rem;
}
.masthead h1 { font-size: 2.35rem; font-weight: 700; margin: 0 0 0.5rem 0; line-height: 1.1; }
.masthead .sub { color: var(--muted); font-size: 0.97rem; max-width: 62ch; margin: 0; }

/* ---------- section headers ---------- */
.section { display: flex; align-items: baseline; gap: 0.85rem; margin: 2.4rem 0 1.1rem 0; }
.section .idx {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--accent);
  border: 1px solid var(--line); border-radius: 3px; padding: 0.12rem 0.42rem;
}
.section h2 { font-size: 1.25rem; margin: 0; font-weight: 600; }
.section .hint { color: var(--muted); font-size: 0.85rem; margin-left: auto; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: #0A1114; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
.side-title {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--muted); margin: 1.5rem 0 0.6rem 0;
}
.side-title:first-child { margin-top: 0; }
.pill {
  display: inline-flex; align-items: center; gap: 0.5rem; font-family: 'IBM Plex Mono', monospace;
  font-size: 0.78rem; padding: 0.35rem 0.7rem; border-radius: 999px;
  border: 1px solid rgba(95,211,196,0.35); background: rgba(95,211,196,0.08); color: var(--accent);
}
.pill.err { border-color: rgba(255,92,108,0.4); background: rgba(255,92,108,0.08); color: var(--malignant); }
.dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.spec { border-top: 1px solid var(--line); margin-top: 0.9rem; }
.spec div {
  display: flex; justify-content: space-between; gap: 1rem; padding: 0.42rem 0;
  border-bottom: 1px solid var(--line); font-size: 0.8rem;
}
.spec .k { color: var(--muted); }
.spec .v { font-family: 'IBM Plex Mono', monospace; color: var(--text); }

/* ---------- readout (prediction) ---------- */
.readout {
  border: 1px solid var(--line); border-radius: 10px; background: var(--panel);
  padding: 1.5rem 1.7rem 1.3rem 1.7rem;
}
.readout .head { display: flex; align-items: flex-end; justify-content: space-between; flex-wrap: wrap; gap: 0.8rem; }
.readout .verdict { font-family: 'Space Grotesk', sans-serif; font-size: 2.5rem; font-weight: 700; line-height: 1; }
.readout .label {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--muted); display: block; margin-bottom: 0.45rem;
}
.readout .conf { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; color: var(--text); }
.readout .conf small { color: var(--muted); font-size: 0.72rem; letter-spacing: 0.12em; display: block; text-align: right; }
.strip { display: flex; height: 12px; border-radius: 3px; overflow: hidden; margin: 1.5rem 0 0.9rem 0; background: #0A1114; }
.strip span { display: block; height: 100%; }
.legend { display: flex; flex-wrap: wrap; gap: 1.6rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; }
.legend .item { display: flex; align-items: center; gap: 0.5rem; color: var(--muted); }
.legend .swatch { width: 9px; height: 9px; border-radius: 2px; }
.legend .val { color: var(--text); }
.truth {
  margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid var(--line);
  font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: var(--muted);
}
.truth b { color: var(--text); font-weight: 500; }

/* ---------- sample gallery ---------- */
.tag {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.1em;
  text-transform: uppercase; padding: 0.12rem 0.45rem; border-radius: 3px; border: 1px solid;
}
.case-id { font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; color: var(--muted); }

[data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] { gap: 0.55rem; }
div[data-testid="stImage"] img { border-radius: 6px; }

/* ---------- controls ---------- */
.stButton > button {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; letter-spacing: 0.04em;
  border-radius: 6px; border: 1px solid var(--line); background: transparent; color: var(--text);
  transition: border-color .15s ease, color .15s ease;
}
.stButton > button:hover { border-color: var(--accent); color: var(--accent); }
.stButton > button[kind="primary"] { background: var(--accent); color: #06110F; border-color: var(--accent); font-weight: 600; }
.stButton > button[kind="primary"]:hover { background: #7CE3D6; color: #06110F; }
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--accent) !important; }
[data-testid="stFileUploaderDropzone"] { background: var(--panel); border: 1px dashed var(--line); }

.caption { color: var(--muted); font-size: 0.85rem; }
.footnote {
  margin-top: 3.5rem; padding-top: 1rem; border-top: 1px solid var(--line);
  color: #5E727B; font-size: 0.75rem; font-family: 'IBM Plex Mono', monospace;
  display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.6rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def section(idx, title, hint=""):
    st.markdown(
        f'<div class="section"><span class="idx">{idx}</span><h2>{title}</h2>'
        f'<span class="hint">{hint}</span></div>',
        unsafe_allow_html=True,
    )


def show_image(img, caption=None):
    """st.image across Streamlit versions (use_container_width vs use_column_width)."""
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img, caption=caption, use_column_width=True)


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
    # every image produces the same feature length, regardless of its content.
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
    img = np.array(pil_image.convert("L"))
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    img_enhanced = _clahe.apply(img_blur)
    img_resized = cv2.resize(img_enhanced, TARGET_SHAPE, interpolation=cv2.INTER_AREA)
    return img_resized.astype(np.float32) / 255.0


def build_pseudo3d_tensor(prev_img, curr_img, next_img):
    """Three PIL images (prev, curr, next) -> (1,3,224,224) tensor + center slice [0,1]."""
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
        self._handles = []
        self._hook_layers()

    def _hook_layers(self):
        def forward_hook(module, inp, output):
            self.features = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def close(self):
        # The model is cached across reruns, so hooks must be released or they
        # accumulate on every inference.
        for h in self._handles:
            h.remove()
        self._handles = []

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
    """center_slice_01: float32 [0,1] grayscale array. Returns RGB uint8 images."""
    img_gray = (center_slice_01 * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlayed = cv2.addWeighted(heatmap_color, alpha, img_rgb, 1 - alpha, 0)
    return img_rgb, heatmap_color, overlayed


# =====================================================================
# 5. MODEL LOADING
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    weights_path = os.path.join(tempfile.gettempdir(), "dual_stream_model.pth")
    if not os.path.exists(weights_path):
        with requests.get(MODEL_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(weights_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    state_dict = torch.load(weights_path, map_location="cpu")

    # Infer the handcrafted feature dimension straight from the checkpoint
    # rather than recomputing it locally — robust to any drift between this
    # app's HOG/LBP params and whatever the model was trained with.
    handcrafted_dim = state_dict["handcrafted_projector.0.weight"].shape[1]

    model = DualStreamPseudo3DVGG(handcrafted_dim=handcrafted_dim, num_classes=len(CLASS_NAMES))
    model.load_state_dict(state_dict)
    model.eval()
    return model, handcrafted_dim


# =====================================================================
# 6. SAMPLE STUDIES (pulled from the project repository)
# =====================================================================
def label_from_filename(name):
    low = name.lower()
    for cname in CLASS_NAMES:
        if cname.lower() in low:
            return cname
    return None


def case_id_from_filename(name):
    nums = re.findall(r"\((\d+)\)", name)
    return f"case {nums[0]}" if nums else os.path.splitext(name)[0]


@st.cache_data(ttl=3600, show_spinner=False)
def list_sample_studies():
    """Filenames in the repo's sample folder, ordered Normal -> Benign -> Malignant."""
    names = []
    try:
        r = requests.get(GH_API_URL, timeout=10)
        if r.status_code == 200:
            names = [
                item["name"] for item in r.json()
                if item.get("type") == "file"
                and item["name"].lower().endswith((".png", ".jpg", ".jpeg"))
            ]
    except Exception:
        names = []

    if not names:
        names = list(FALLBACK_SAMPLES)

    def sort_key(n):
        lab = label_from_filename(n)
        return (CLASS_NAMES.index(lab) if lab in CLASS_NAMES else 99, n)

    return sorted(names, key=sort_key)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sample_bytes(name):
    url = GH_RAW_BASE + urllib.parse.quote(name)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content


# =====================================================================
# 7. SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown('<div class="side-title">Model</div>', unsafe_allow_html=True)
    slot = st.empty()
    slot.markdown(
        '<span class="pill"><span class="dot"></span>Loading weights…</span>',
        unsafe_allow_html=True,
    )
    try:
        model, handcrafted_dim = load_model()
        slot.markdown(
            '<span class="pill"><span class="dot"></span>Weights loaded</span>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        slot.markdown(
            '<span class="pill err"><span class="dot"></span>Weights unavailable</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"{type(e).__name__}: {e}")
        st.stop()

    st.markdown(
        f"""
        <div class="spec">
          <div><span class="k">Architecture</span><span class="v">Dual-stream</span></div>
          <div><span class="k">Spatial stream</span><span class="v">VGG16, pseudo-3D</span></div>
          <div><span class="k">Texture stream</span><span class="v">HOG + LBP</span></div>
          <div><span class="k">Feature vector</span><span class="v">{handcrafted_dim:,}-d</span></div>
          <div><span class="k">Fusion head</span><span class="v">512 + 256 → 256 → 3</span></div>
          <div><span class="k">Input</span><span class="v">3 × 224 × 224</span></div>
          <div><span class="k">Explainability</span><span class="v">Grad-CAM</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-title">Classes</div>', unsafe_allow_html=True)
    for cname in CLASS_NAMES:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:.6rem;padding:.25rem 0;'
            f'font-family:IBM Plex Mono,monospace;font-size:.82rem;">'
            f'<span style="width:9px;height:9px;border-radius:2px;background:{CLASS_COLORS[cname]}"></span>'
            f'{cname}</div>',
            unsafe_allow_html=True,
        )


# =====================================================================
# 8. MASTHEAD
# =====================================================================
st.markdown(
    """
    <div class="masthead">
      <span class="eyebrow">Dual-stream pseudo-3D convolutional network</span>
      <h1>Lung Cancer Detection</h1>
      <p class="sub">A VGG16 spatial stream over three consecutive CT slices, fused with a
      HOG/LBP texture descriptor, classifying each study as normal, benign or malignant —
      with Grad-CAM evidence for every prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# 9. STUDY SELECTION
# =====================================================================
if "sample_choice" not in st.session_state:
    st.session_state.sample_choice = None

section("01", "Select a study", "sample scans from the project dataset, or your own upload")

tab_samples, tab_upload = st.tabs(["Sample studies", "Upload a scan"])

prev_img = curr_img = next_img = None
source_label = None
reference_label = None

with tab_samples:
    samples = list_sample_studies()
    st.markdown(
        '<p class="caption">Twelve held-out scans, four per class. '
        'Select one to run it through the model.</p>',
        unsafe_allow_html=True,
    )

    n_cols = 4
    for row_start in range(0, len(samples), n_cols):
        cols = st.columns(n_cols, gap="medium")
        for col, name in zip(cols, samples[row_start:row_start + n_cols]):
            with col:
                with st.container(border=True):
                    lab = label_from_filename(name) or "Unlabelled"
                    color = CLASS_COLORS.get(lab, "#8CA0A9")
                    try:
                        show_image(fetch_sample_bytes(name))
                    except Exception:
                        st.markdown('<p class="caption">Preview unavailable</p>', unsafe_allow_html=True)

                    st.markdown(
                        f'<div style="display:flex;align-items:center;justify-content:space-between;gap:.5rem;">'
                        f'<span class="tag" style="color:{color};border-color:{color}44;'
                        f'background:{color}14;">{lab}</span>'
                        f'<span class="case-id">{case_id_from_filename(name)}</span></div>',
                        unsafe_allow_html=True,
                    )

                    selected = st.session_state.sample_choice == name
                    if st.button(
                        "Selected" if selected else "Select",
                        key=f"pick_{name}",
                        type="primary" if selected else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state.sample_choice = name
                        st.rerun()

with tab_upload:
    mode = st.radio(
        "Input format",
        ["Single slice", "Three consecutive slices"],
        horizontal=True,
        help=(
            "The model was trained on triplets of adjacent CT slices. A single image is "
            "duplicated across all three channels — a reasonable approximation, though "
            "accuracy is highest on true adjacent slices."
        ),
    )

    if mode == "Single slice":
        uploaded = st.file_uploader("Lung CT slice", type=["png", "jpg", "jpeg"], key="single")
        if uploaded is not None:
            curr_img = Image.open(io.BytesIO(uploaded.read()))
            prev_img = next_img = curr_img
            source_label = uploaded.name
            st.session_state.sample_choice = None
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            f_prev = st.file_uploader("Previous slice", type=["png", "jpg", "jpeg"], key="prev")
        with c2:
            f_curr = st.file_uploader("Centre slice", type=["png", "jpg", "jpeg"], key="curr")
        with c3:
            f_next = st.file_uploader("Next slice", type=["png", "jpg", "jpeg"], key="next")
        if f_prev and f_curr and f_next:
            prev_img = Image.open(io.BytesIO(f_prev.read()))
            curr_img = Image.open(io.BytesIO(f_curr.read()))
            next_img = Image.open(io.BytesIO(f_next.read()))
            source_label = f_curr.name
            st.session_state.sample_choice = None

# Sample selection is used when nothing has been uploaded.
if curr_img is None and st.session_state.sample_choice:
    name = st.session_state.sample_choice
    try:
        curr_img = Image.open(io.BytesIO(fetch_sample_bytes(name)))
        prev_img = next_img = curr_img
        source_label = name
        reference_label = label_from_filename(name)
    except Exception as e:
        st.error(f"Could not load {name} — {type(e).__name__}")

# ---- run bar ----
try:
    left, right = st.columns([3, 1], vertical_alignment="bottom")
except TypeError:  # older Streamlit builds have no vertical_alignment
    left, right = st.columns([3, 1])
with left:
    if source_label:
        st.markdown(
            f'<p class="caption" style="margin:1.4rem 0 0 0;">Ready · '
            f'<span style="font-family:IBM Plex Mono,monospace;color:#E3EBEE;">{source_label}</span></p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="caption" style="margin:1.4rem 0 0 0;">No study selected.</p>',
            unsafe_allow_html=True,
        )
with right:
    run = st.button(
        "Run analysis",
        type="primary",
        disabled=curr_img is None,
        use_container_width=True,
    )

# =====================================================================
# 10. INFERENCE + RESULTS
# =====================================================================
if curr_img is not None and run:
    with st.spinner("Preprocessing, inference and Grad-CAM…"):
        img_tensor, center_slice_01 = build_pseudo3d_tensor(prev_img, curr_img, next_img)
        handcrafted_vec = extract_handcrafted_features(center_slice_01).unsqueeze(0)

        cam_engine = DualStreamGradCAM(model, model.vgg_features[-1])
        try:
            img_tensor.requires_grad_(False)
            with torch.no_grad():
                logits = model(img_tensor, handcrafted_vec)
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = int(torch.argmax(probs).item())
                pred_name = CLASS_NAMES[pred_idx]
                confidence = float(probs[pred_idx].item()) * 100

            with torch.set_grad_enabled(True):
                heatmap, _ = cam_engine.generate_heatmap(
                    img_tensor, handcrafted_vec, target_class=pred_idx
                )
        finally:
            cam_engine.close()

        img_rgb, heatmap_color, overlay = overlay_heatmap(center_slice_01, heatmap, alpha=0.4)

    p = [float(probs[i].item()) * 100 for i in range(len(CLASS_NAMES))]

    section("02", "Result", "softmax over the fused representation")

    segments = "".join(
        f'<span style="width:{p[i]:.4f}%;background:{CLASS_COLORS[c]};"></span>'
        for i, c in enumerate(CLASS_NAMES)
    )
    legend = "".join(
        f'<div class="item"><span class="swatch" style="background:{CLASS_COLORS[c]}"></span>'
        f'{c} <span class="val">{p[i]:.2f}%</span></div>'
        for i, c in enumerate(CLASS_NAMES)
    )

    truth_html = ""
    if reference_label:
        agree = reference_label == pred_name
        mark = "agrees with" if agree else "differs from"
        mcol = CLASS_COLORS["Normal"] if agree else CLASS_COLORS["Malignant"]
        truth_html = (
            f'<div class="truth">Dataset label <b>{reference_label}</b> — prediction '
            f'<span style="color:{mcol}">{mark}</span> the reference.</div>'
        )

    st.markdown(
        f"""
        <div class="readout">
          <div class="head">
            <div>
              <span class="label">Predicted class</span>
              <span class="verdict" style="color:{CLASS_COLORS[pred_name]}">{pred_name}</span>
            </div>
            <div class="conf">{confidence:.2f}%<small>Confidence</small></div>
          </div>
          <div class="strip">{segments}</div>
          <div class="legend">{legend}</div>
          {truth_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    section("03", "Visual evidence", "Grad-CAM, final VGG16 convolutional block")
    st.markdown(
        f'<p class="caption">Regions weighted most heavily in the '
        f'<span style="color:{CLASS_COLORS[pred_name]}">{pred_name.lower()}</span> decision.</p>',
        unsafe_allow_html=True,
    )

    v1, v2, v3 = st.columns(3, gap="medium")
    with v1:
        with st.container(border=True):
            show_image(img_rgb, caption="Preprocessed centre slice")
    with v2:
        with st.container(border=True):
            show_image(heatmap_color, caption="Activation map")
    with v3:
        with st.container(border=True):
            show_image(overlay, caption="Overlay")

    # Downloadable figure, styled to match the interface
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor="#0B1013")
    for ax, im, title, cmap in [
        (axes[0], img_rgb, "Centre slice", None),
        (axes[1], heatmap, "Grad-CAM", "jet"),
        (axes[2], overlay, f"{pred_name} · {confidence:.1f}%", None),
    ]:
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, color="#E3EBEE", fontsize=12, pad=12)
        ax.set_facecolor("#0B1013")
        ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="#0B1013")
    plt.close(fig)

    d1, _ = st.columns([1, 3])
    with d1:
        st.download_button(
            "Download figure",
            data=buf.getvalue(),
            file_name=f"gradcam_{pred_name.lower()}.png",
            mime="image/png",
            use_container_width=True,
        )

elif curr_img is None:
    st.markdown(
        '<p class="caption" style="margin-top:2rem;">Choose a sample study above or upload a scan, '
        'then run the analysis.</p>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footnote"><span>Dual-stream pseudo-3D VGG16 · HOG + LBP fusion · Grad-CAM</span>'
    '<span>Research prototype — not for clinical use</span></div>',
    unsafe_allow_html=True,
)
