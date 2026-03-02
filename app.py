import io
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn

from transformers import Dinov2Model, AutoImageProcessor

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model_simpledinov2.pth"
BACKBONE = "facebook/dinov2-small"
IMAGE_SIZE = 128
NUM_CLASSES = 2

st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌾",
    layout="wide"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.block-container {
    padding-top: 2rem;
}
.card {
    background-color: #161B22;
    padding: 1rem;
    border-radius: 12px;
    text-align: center;
}
.title {
    font-size: 40px;
    font-weight: 700;
    color: white;
}
.subtitle {
    font-size: 18px;
    color: #9BA3AF;
}
.status-ok {
    color: #22c55e;
    font-weight: 600;
}
.status-bad {
    color: #ef4444;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🌾 AgriVision AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Satellite Crop Segmentation using DinoV2</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")

overlay_strength = st.sidebar.slider("Overlay Opacity", 0.1, 1.0, 0.5)
run_button = st.sidebar.button("🚀 Run Inference")

# ---------------- MODEL ----------------
class SimpleDinov2Seg(nn.Module):
    def __init__(self, backbone_name=BACKBONE, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(backbone_name)
        hidden_dim = self.backbone.config.hidden_size
        patch = self.backbone.config.patch_size

        self.patch = patch
        self.hidden = hidden_dim

        self.conv = nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim//4, hidden_dim//8, 2, stride=2),
            nn.ReLU()
        )
        self.final = nn.Conv2d(hidden_dim//8, num_classes, 1)

    def forward(self, x):
        out = self.backbone(pixel_values=x)["last_hidden_state"]
        seq = out.shape[1]
        expected = (x.shape[2]//self.patch)*(x.shape[3]//self.patch)

        if seq == expected+1:
            out = out[:,1:,:]

        H = x.shape[2]//self.patch
        W = x.shape[3]//self.patch

        out = out.permute(0,2,1).contiguous().view(x.shape[0], self.hidden, H, W)
        out = self.conv(out)
        out = self.up(out)
        out = self.final(out)

        out = nn.functional.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return out

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(BACKBONE)
    model = SimpleDinov2Seg()

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)

    model.to(DEVICE)
    model.eval()
    return model, processor

try:
    model, processor = load_model()
    st.sidebar.markdown('<div class="status-ok">✔ Model Loaded</div>', unsafe_allow_html=True)
except:
    st.sidebar.markdown('<div class="status-bad">✖ Model Missing</div>', unsafe_allow_html=True)
    st.stop()

# ---------------- UPLOAD ----------------
uploaded = st.file_uploader("Upload Satellite Image", type=["jpg","png","tif"])

def preprocess(img):
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img)/255.0
    t = torch.tensor(arr).permute(2,0,1).unsqueeze(0).float()

    mean = torch.tensor(processor.image_mean).view(1,3,1,1)
    std = torch.tensor(processor.image_std).view(1,3,1,1)
    return (t-mean)/std

def overlay(img, mask):
    img = np.array(img)
    red = np.zeros_like(img)
    red[:,:,0] = 255
    img = img*(1-overlay_strength)
    img[mask==1] = red[mask==1]*overlay_strength
    return img.astype(np.uint8)

# ---------------- INFERENCE ----------------
if uploaded and run_button:

    with st.spinner("Running AI segmentation..."):
        image = Image.open(uploaded).convert("RGB")
        inp = preprocess(image).to(DEVICE)

        with torch.no_grad():
            logits = model(inp)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = probs.argmax(axis=0)

        conf = np.mean(probs.max(axis=0))

        mask = (pred*255).astype(np.uint8)
        overlay_img = overlay(image.resize((IMAGE_SIZE, IMAGE_SIZE)), pred)

    # ---------------- DISPLAY ----------------
    c1,c2,c3 = st.columns(3)

    with c1:
        st.markdown("### Input")
        st.image(image, use_column_width=True)

    with c2:
        st.markdown("### Prediction")
        st.image(mask, use_column_width=True)

    with c3:
        st.markdown("### Overlay")
        st.image(overlay_img, use_column_width=True)

    st.markdown("---")
    st.metric("Model Confidence", f"{conf*100:.2f}%")

    # download
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    st.download_button("Download Mask", buf.getvalue(), "mask.png")