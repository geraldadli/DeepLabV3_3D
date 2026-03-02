# app.py
import io
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn

# transformers (Dinov2 + processor)
from transformers import Dinov2Model, AutoImageProcessor

# rasterio for GeoTIFF handling (optional but strongly recommended)
try:
    import rasterio
    from rasterio.io import MemoryFile
except Exception:
    rasterio = None
    MemoryFile = None

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model_simpledinov2.pth"  # place your model in repo root
BACKBONE = "facebook/dinov2-small"
IMAGE_SIZE = 128
NUM_CLASSES = 2

st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌾",
    layout="wide"
)

# ---------------- STYLE ----------------
st.markdown(
    """
<style>
.main { background-color: #0E1117; color: #E6EEF3; }
.block-container { padding-top: 2rem; }
.card { background-color: #161B22; padding: 1rem; border-radius: 12px; text-align: center; }
.title { font-size: 40px; font-weight: 700; color: white; }
.subtitle { font-size: 18px; color: #9BA3AF; }
.status-ok { color: #22c55e; font-weight: 600; }
.status-bad { color: #ef4444; font-weight: 600; }
small { color: #9BA3AF; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🌾 AgriVision AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Satellite Crop Segmentation using DinoV2 — Robust Reader & UI</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")
overlay_strength = st.sidebar.slider("Overlay Opacity", 0.1, 1.0, 0.5)
run_button = st.sidebar.button("🚀 Run Inference")
st.sidebar.markdown("### Model")
st.sidebar.write(f"Checkpoint: `{MODEL_PATH}`")
st.sidebar.write(f"Device: `{DEVICE}`")
st.sidebar.markdown("---")

# ---------------- MODEL DEFINITION ----------------
class SimpleDinov2Seg(nn.Module):
    def __init__(self, backbone_name=BACKBONE, num_classes=NUM_CLASSES):
        super().__init__()
        # make sure transformers and model are available in deployment environment
        self.backbone = Dinov2Model.from_pretrained(backbone_name)
        hidden_dim = self.backbone.config.hidden_size
        patch = getattr(self.backbone.config, "patch_size", 16)

        self.patch = patch
        self.hidden = hidden_dim

        self.conv = nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.final = nn.Conv2d(hidden_dim // 8, num_classes, kernel_size=1)

    def forward(self, x):
        # x: B,C,H,W
        out = self.backbone(pixel_values=x)["last_hidden_state"]  # B, tokens, hidden
        seq = out.shape[1]
        expected = (x.shape[2] // self.patch) * (x.shape[3] // self.patch)
        if seq == expected + 1:
            out = out[:, 1:, :]

        H = x.shape[2] // self.patch
        W = x.shape[3] // self.patch

        out = out.permute(0, 2, 1).contiguous().view(x.shape[0], self.hidden, H, W)
        out = self.conv(out)
        out = self.up(out)
        out = self.final(out)
        out = nn.functional.interpolate(out, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        return out

# ---------------- MODEL + PROCESSOR LOADER ----------------
@st.cache_resource(show_spinner=False)
def load_model_and_processor(model_path: str = MODEL_PATH, backbone: str = BACKBONE):
    # Attempt to load HF processor (for normalization); fallback to simple norm if unavailable
    processor = None
    try:
        processor = AutoImageProcessor.from_pretrained(backbone)
    except Exception:
        processor = None

    # instantiate model and load checkpoint
    model = SimpleDinov2Seg(backbone_name=backbone, num_classes=NUM_CLASSES)
    ckpt_path = Path(model_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model file not found at {ckpt_path}. Please put {model_path} in repo root or update MODEL_PATH.")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    # accept nested dict with 'model_state' or raw state dict
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model, processor

# load and show status
try:
    model, processor = load_model_and_processor()
    st.sidebar.markdown('<div class="status-ok">✔ Model Loaded</div>', unsafe_allow_html=True)
except Exception as e:
    st.sidebar.markdown('<div class="status-bad">✖ Model Missing / Failed</div>', unsafe_allow_html=True)
    st.sidebar.error(str(e))
    st.stop()

# ---------------- Robust file reader ----------------
def read_uploaded_image(uploaded_file, image_size=IMAGE_SIZE):
    """
    Return (pil_rgb_resized, rasterio_profile_or_None)
    Raises RuntimeError on failure with a helpful message.
    """
    if uploaded_file is None:
        raise RuntimeError("No file provided.")

    filename = getattr(uploaded_file, "name", "")
    raw = uploaded_file.read()
    if raw is None or len(raw) == 0:
        raise RuntimeError("Empty file or failed to read upload.")

    lower = filename.lower()

    # Prefer rasterio for .tif/.tiff
    if lower.endswith((".tif", ".tiff")) and rasterio is not None and MemoryFile is not None:
        try:
            with MemoryFile(raw) as mem:
                with mem.open() as src:
                    arr = src.read(out_dtype="float32")  # (C,H,W)
                    profile = src.profile.copy()
                    # take up to 3 bands (replicate if fewer)
                    if arr.shape[0] >= 3:
                        img_arr = arr[:3]
                    else:
                        reps = int(np.ceil(3.0 / max(1, arr.shape[0])))
                        stacked = np.tile(arr, (reps, 1, 1))
                        img_arr = stacked[:3]
                    # per-image min-max normalize
                    mn = img_arr.min()
                    mx = img_arr.max()
                    if mx - mn < 1e-8:
                        img_hwc = np.moveaxis(np.clip(img_arr, 0, 1), 0, -1).astype("float32")
                    else:
                        img_hwc = np.moveaxis((img_arr - mn) / (mx - mn + 1e-6), 0, -1).astype("float32")
                    pil = Image.fromarray((np.clip(img_hwc, 0, 1) * 255).astype("uint8"))
                    pil = pil.convert("RGB")
                    pil = pil.resize((image_size, image_size), resample=Image.BILINEAR)
                    return pil, profile
        except Exception as e:
            raise RuntimeError(f"Failed reading TIFF with rasterio: {e}")

    # Otherwise try PIL for common formats
    try:
        bio = io.BytesIO(raw)
        pil = Image.open(bio)
        pil = pil.convert("RGB")
        pil = pil.resize((image_size, image_size), resample=Image.BILINEAR)
        return pil, None
    except UnidentifiedImageError as e:
        raise RuntimeError("Uploaded file is not a recognized image format. If you uploaded a GeoTIFF, ensure it's valid .tif/.tiff. "
                           f"Original error: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to open uploaded file: {e}")

# ---------------- Preprocess helper ----------------
def pil_to_input_tensor(pil: Image.Image, proc):
    """
    Convert PIL to 1xCxHxW tensor and normalize (returns tensor on CPU).
    Use HF processor mean/std if available, else simple (x-0.5)/0.5 scaling.
    """
    arr = np.array(pil).astype("float32") / 255.0  # H,W,3
    tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).float()  # 1,C,H,W
    if proc is not None and hasattr(proc, "image_mean") and hasattr(proc, "image_std"):
        mean = torch.tensor(proc.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(proc.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        tensor = (tensor - mean) / (std + 1e-9)
    else:
        tensor = (tensor - 0.5) / 0.5
    return tensor

# ---------------- Visualization helpers ----------------
def mask_to_uint8(mask_arr):
    # mask_arr: H,W ints in {0,...}
    return (mask_arr.astype(np.uint8) * 255)

def colored_overlay(pil_img, mask_arr, alpha=0.5):
    base = np.array(pil_img).astype(np.float32)
    overlay_color = np.zeros_like(base, dtype=np.uint8)
    overlay_color[mask_arr == 1] = [255, 0, 0]
    # blend
    blended = (base * (1 - alpha) + overlay_color.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return blended

# ---------------- UPLOAD UI ----------------
uploaded = st.file_uploader("Upload Satellite Image (GeoTIFF or JPG/PNG)", type=["tif", "tiff", "jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a GeoTIFF (.tif) or JPG/PNG to run segmentation.")
else:
    # show filename + size
    st.markdown(f"**File:** {getattr(uploaded, 'name', '')} — **Size:** {uploaded.size / 1024:.1f} KB")
    if run_button:
        # run inference
        try:
            with st.spinner("Reading file..."):
                pil_img, tif_profile = read_uploaded_image(uploaded, image_size=IMAGE_SIZE)
        except Exception as e:
            st.error("Failed to read uploaded file.")
            st.exception(e)
            st.stop()

        # preprocess
        try:
            input_tensor = pil_to_input_tensor(pil_img, processor).to(DEVICE)
        except Exception as e:
            st.error("Failed to preprocess image.")
            st.exception(e)
            st.stop()

        # inference
        try:
            with torch.no_grad():
                logits = model(input_tensor)  # B,C,H,W
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # C,H,W
                pred = np.argmax(probs, axis=0).astype(np.uint8)       # H,W
                conf_score = float(np.mean(np.max(probs, axis=0)))    # mean per-pixel max prob
        except Exception as e:
            st.error("Inference failed.")
            st.exception(e)
            st.stop()

        # visuals
        mask_vis = mask_to_uint8(pred)     # H,W uint8 (0/255)
        overlay_img = colored_overlay(pil_img, pred, alpha=overlay_strength)

        # display in columns
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            st.markdown("### Input (resized)")
            st.image(pil_img, use_column_width=True)
        with c2:
            st.markdown("### Predicted Mask")
            st.image(mask_vis, use_column_width=True, clamp=True)
        with c3:
            st.markdown("### Overlay (red = class 1)")
            st.image(overlay_img, use_column_width=True)

        st.markdown("---")
        st.metric("Model mean confidence", f"{conf_score * 100:.2f}%")

        # Download options
        # PNG mask
        buf = io.BytesIO()
        Image.fromarray(mask_vis).save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download mask (PNG)", data=buf.getvalue(), file_name="pred_mask.png", mime="image/png")

        # GeoTIFF export (if original profile present and rasterio available)
        if tif_profile is not None and rasterio is not None:
            try:
                # build output array as uint8 0/255 and scale back to original dims if profile available
                out_mask = (pred.astype(np.uint8) * 255)
                orig_h = tif_profile.get("height", None)
                orig_w = tif_profile.get("width", None)
                if orig_h and orig_w and (orig_h != out_mask.shape[0] or orig_w != out_mask.shape[1]):
                    # resize back using nearest neighbor
                    pil_mask = Image.fromarray(out_mask)
                    pil_mask = pil_mask.resize((orig_w, orig_h), resample=Image.NEAREST)
                    out_arr = np.array(pil_mask).astype(np.uint8)
                else:
                    out_arr = out_mask.astype(np.uint8)

                out_profile = tif_profile.copy()
                out_profile.update(dtype=rasterio.uint8, count=1, compress="lzw")

                tmp_path = "pred_mask_georef.tif"
                with rasterio.open(tmp_path, "w", **out_profile) as dst:
                    dst.write(out_arr[np.newaxis, :, :].astype(rasterio.uint8))
                with open(tmp_path, "rb") as f:
                    st.download_button("Download mask (GeoTIFF)", data=f.read(), file_name="pred_mask_georef.tif", mime="image/tiff")
            except Exception as e:
                st.warning(f"GeoTIFF export failed: {e}")
                st.exception(e)