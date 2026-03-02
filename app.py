# app.py
import io
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# optional libs
try:
    import rasterio
    from rasterio.io import MemoryFile
except Exception:
    rasterio = None
    MemoryFile = None

try:
    import tifffile
except Exception:
    tifffile = None

try:
    import imageio.v3 as iio
except Exception:
    iio = None

try:
    import scipy.ndimage as ndi
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# transformers processor/backbone
try:
    from transformers import Dinov2Model, AutoImageProcessor
    HF_AVAILABLE = True
except Exception:
    Dinov2Model = None
    AutoImageProcessor = None
    HF_AVAILABLE = False

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model_simpledinov2.pth"   # put your checkpoint here
BACKBONE = "facebook/dinov2-small"
IMAGE_SIZE = 128
NUM_CLASSES = 2

st.set_page_config(page_title="AgriVision AI (prod)", layout="wide")
st.title("🌾 AgriVision AI — Robust Segmentation + Diagnostics")
st.write("Upload GeoTIFF/JPEG/PNG. Use sidebar to control preprocessing, thresholding & cleanup.")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Model & Preprocessing")
st.sidebar.write(f"Device: `{DEVICE}`")
st.sidebar.write(f"Checkpoint: `{MODEL_PATH}`")
norm_choice = st.sidebar.selectbox("Normalization", options=[
    "auto (hf processor if available)", "imagenet (0.485/0.456/0.406)", "centered 0.5 (x-0.5)/0.5"
], index=0)
apply_percentile = st.sidebar.checkbox("Percentile stretch for TIFF (2-98)", value=True)
percentile_low = st.sidebar.slider("Percentile low (for stretch)", 0, 10, 2)
percentile_high = st.sidebar.slider("Percentile high (for stretch)", 90, 100, 98)

st.sidebar.markdown("---")
st.sidebar.header("Postprocessing / Debug")
show_prob = st.sidebar.checkbox("Show probability heatmap & histogram", value=True)
threshold = st.sidebar.slider("Prob threshold for class1", 0.01, 0.99, 0.5, 0.01)
min_area = st.sidebar.slider("Min connected component area (px)", 0, 5000, 50, 10)
apply_cleanup = st.sidebar.checkbox("Remove small objects (connected components)", value=True)
overlay_alpha = st.sidebar.slider("Overlay alpha", 0.0, 1.0, 0.5)
st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Inference")

# ---------------- Model definition ----------------
class SimpleDinov2Seg(nn.Module):
    def __init__(self, backbone_name=BACKBONE, num_classes=NUM_CLASSES):
        super().__init__()
        if Dinov2Model is None:
            raise RuntimeError("Dinov2Model from transformers not available in environment.")
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
        out = self.backbone(pixel_values=x)["last_hidden_state"]
        seq = out.shape[1]
        expected = (x.shape[2] // self.patch) * (x.shape[3] // self.patch)
        if seq == expected + 1:
            out = out[:, 1:, :]
        H = x.shape[2] // self.patch
        W = x.shape[3] // self.patch
        out = out.permute(0,2,1).contiguous().view(x.shape[0], self.hidden, H, W)
        out = self.conv(out)
        out = self.up(out)
        out = self.final(out)
        out = nn.functional.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return out

# ---------------- Load model & processor ----------------
@st.cache_resource(show_spinner=False)
def load_model_and_processor(model_path=MODEL_PATH, backbone=BACKBONE):
    # processor if available
    processor = None
    if HF_AVAILABLE and AutoImageProcessor is not None:
        try:
            processor = AutoImageProcessor.from_pretrained(backbone)
        except Exception:
            processor = None
    # instantiate model architecture
    model = SimpleDinov2Seg(backbone_name=backbone, num_classes=NUM_CLASSES)
    ckpt_path = Path(model_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Please upload to repo root.")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and ("model_state" in ckpt or "state_dict" in ckpt):
        # common variants
        state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt
    # load tolerant
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model, processor, state

try:
    model, processor, ckpt_state = load_model_and_processor()
    st.sidebar.success("Model loaded")
    # debug key summary
    st.sidebar.write("checkpoint keys (sample):", list(ckpt_state.keys())[:10])
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

# ---------------- Robust reader ----------------
def read_uploaded_image(uploaded_file, image_size=IMAGE_SIZE,
                        percentile_low=2, percentile_high=98, apply_percentile=True):
    """Return (PIL RGB resized, rasterio_profile_or_None, debug_info)"""
    if uploaded_file is None:
        raise RuntimeError("No file provided.")
    filename = getattr(uploaded_file, "name", "")
    raw = uploaded_file.read()
    if raw is None or len(raw) == 0:
        raise RuntimeError("Empty upload.")
    lower = filename.lower()
    debug = {}
    # TIFF branch
    if lower.endswith((".tif", ".tiff")):
        # prefer rasterio
        if rasterio is not None and MemoryFile is not None:
            try:
                with MemoryFile(raw) as mem:
                    with mem.open() as src:
                        arr = src.read(out_dtype='float32')  # C,H,W
                        profile = src.profile.copy()
                        debug['reader'] = 'rasterio'
                        debug['orig_shape'] = arr.shape
                        # select first 3 bands or replicate
                        if arr.shape[0] >= 3:
                            rgb = arr[:3]
                        else:
                            rgb = np.tile(arr, (3 // arr.shape[0] + 1, 1, 1))[:3]
                        # percentile stretch
                        if apply_percentile:
                            p_low = np.nanpercentile(rgb, percentile_low)
                            p_high = np.nanpercentile(rgb, percentile_high)
                            rgb = np.clip((rgb - p_low) / (p_high - p_low + 1e-9), 0, 1)
                        else:
                            # simple min-max
                            mn = rgb.min(); mx = rgb.max()
                            rgb = (rgb - mn) / (mx - mn + 1e-9)
                        img = np.moveaxis(rgb, 0, -1)  # H,W,3
                        pil = Image.fromarray((np.clip(img,0,1)*255).astype('uint8')).convert("RGB")
                        pil = pil.resize((image_size, image_size), resample=Image.BILINEAR)
                        debug['final_shape'] = pil.size
                        return pil, profile, debug
            except Exception as e:
                debug['rasterio_error'] = str(e)
        # tifffile
        if tifffile is not None:
            try:
                arr = tifffile.imread(io.BytesIO(raw))
                arr = np.asarray(arr)
                debug['reader'] = 'tifffile'
                debug['orig_shape'] = arr.shape
                # tifffile may return H,W or H,W,C or C,H,W
                if arr.ndim == 2:
                    arr = np.stack([arr,arr,arr], axis=-1)
                elif arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[0] != arr.shape[2]:
                    arr = np.moveaxis(arr, 0, -1)
                # use only first 3 channels
                if arr.shape[2] >=3:
                    rgb = arr[:,:, :3].astype('float32')
                else:
                    rgb = np.repeat(arr, 3, axis=2)[:,:,:3].astype('float32')
                if apply_percentile:
                    p_low = np.nanpercentile(rgb, percentile_low)
                    p_high = np.nanpercentile(rgb, percentile_high)
                    rgb = np.clip((rgb - p_low) / (p_high - p_low + 1e-9), 0, 1)
                else:
                    mn = rgb.min(); mx = rgb.max()
                    rgb = (rgb - mn) / (mx - mn + 1e-9)
                pil = Image.fromarray((np.clip(rgb,0,1)*255).astype('uint8')).convert("RGB")
                pil = pil.resize((image_size, image_size), resample=Image.BILINEAR)
                debug['final_shape'] = pil.size
                return pil, None, debug
            except Exception as e:
                debug['tifffile_error'] = str(e)
        # imageio fallback
        if iio is not None:
            try:
                arr = iio.imread(io.BytesIO(raw))
                arr = np.asarray(arr)
                debug['reader'] = 'imageio'
                debug['orig_shape'] = arr.shape
                if arr.ndim == 2:
                    arr = np.stack([arr,arr,arr], axis=-1)
                if arr.shape[2] > 3:
                    arr = arr[:,:,:3]
                if apply_percentile:
                    p_low = np.nanpercentile(arr, percentile_low)
                    p_high = np.nanpercentile(arr, percentile_high)
                    arr = np.clip((arr - p_low) / (p_high - p_low + 1e-9), 0, 1)
                else:
                    mn = arr.min(); mx = arr.max()
                    arr = (arr - mn) / (mx - mn + 1e-9)
                pil = Image.fromarray((np.clip(arr,0,1)*255).astype('uint8')).convert("RGB")
                pil = pil.resize((image_size, image_size), resample=Image.BILINEAR)
                debug['final_shape'] = pil.size
                return pil, None, debug
            except Exception as e:
                debug['imageio_error'] = str(e)
        # if we reach here: no reader succeeded
        raise RuntimeError("Unable to read TIFF. Install rasterio or tifffile in requirements.txt. Details: " + str(debug))
    else:
        # non-tiff: try PIL
        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
            pil = pil.resize((image_size, image_size), resample=Image.BILINEAR)
            return pil, None, {'reader':'PIL', 'final_shape':pil.size}
        except UnidentifiedImageError as e:
            raise RuntimeError("Uploaded file is not a recognized image (PIL). If GeoTIFF, install rasterio or tifffile. Error: " + str(e))
        except Exception as e:
            raise RuntimeError("Failed reading uploaded file: " + str(e))

# ---------------- Preprocess helper ----------------
def pil_to_tensor(pil, proc=None, norm_choice="auto"):
    """Return 1xCxHxW tensor (float32)."""
    arr = np.array(pil).astype('float32') / 255.0  # H,W,3
    tensor = torch.tensor(arr).permute(2,0,1).unsqueeze(0).float()
    if norm_choice == "auto" and proc is not None:
        if hasattr(proc, "image_mean") and hasattr(proc, "image_std"):
            mean = torch.tensor(proc.image_mean, dtype=torch.float32).view(1,3,1,1)
            std  = torch.tensor(proc.image_std, dtype=torch.float32).view(1,3,1,1)
            tensor = (tensor - mean) / (std + 1e-9)
            used = ("hf_processor", proc.image_mean, proc.image_std)
            return tensor, used
    # explicit choices
    if norm_choice == "imagenet":
        mean = torch.tensor([0.485,0.456,0.406], dtype=torch.float32).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225], dtype=torch.float32).view(1,3,1,1)
        tensor = (tensor - mean) / (std + 1e-9)
        used = ("imagenet", [0.485,0.456,0.406], [0.229,0.224,0.225])
        return tensor, used
    # default centered 0.5
    mean = torch.tensor([0.5,0.5,0.5], dtype=torch.float32).view(1,3,1,1)
    std  = torch.tensor([0.5,0.5,0.5], dtype=torch.float32).view(1,3,1,1)
    tensor = (tensor - mean) / (std + 1e-9)
    used = ("center05", [0.5,0.5,0.5], [0.5,0.5,0.5])
    return tensor, used

# ---------------- Simple visualization helpers ----------------
def mask_to_uint8(mask):
    return (mask.astype(np.uint8) * 255)

def colored_overlay(pil_img, mask, alpha=0.5, color=(255,0,0)):
    base = np.array(pil_img).astype(np.float32)
    overlay = np.zeros_like(base, dtype=np.uint8)
    overlay[mask==1] = color
    blended = (base * (1-alpha) + overlay.astype(np.float32) * alpha).clip(0,255).astype(np.uint8)
    return blended

# ---------------- UI: upload + run ----------------
uploaded = st.file_uploader("Upload GeoTIFF (.tif/.tiff) or JPG/PNG", type=["tif","tiff","jpg","jpeg","png"])
if uploaded is None:
    st.info("Upload an image to run inference.")
else:
    st.markdown(f"**File**: `{getattr(uploaded,'name','')}` — size: {uploaded.size/1024:.1f} KB")
    # show reader choice and percentile options
    st.write("Preprocessing options:", f"percentile stretch = {apply_percentile} ({percentile_low}-{percentile_high})",
             f"normalization choice = {norm_choice}")
    if run_button:
        try:
            pil_img, tif_profile, debug = read_uploaded_image(uploaded,
                                                              image_size=IMAGE_SIZE,
                                                              percentile_low=percentile_low,
                                                              percentile_high=percentile_high,
                                                              apply_percentile=apply_percentile)
        except Exception as e:
            st.error("Failed to read uploaded file.")
            st.exception(e)
            st.stop()

        st.write("Reader debug:", debug)
        # preprocess
        try:
            # decide norm mode
            chosen_mode = None
            if norm_choice == "auto":
                chosen_mode = "auto"
            elif norm_choice == "imagenet":
                chosen_mode = "imagenet"
            else:
                chosen_mode = "center05"
            inp_tensor, used_norm = pil_to_tensor(pil_img, proc=processor, norm_choice=chosen_mode)
            st.sidebar.write("Used normalization:", used_norm)
            st.sidebar.write("Input tensor mean/std (per-channel):",
                             [float(inp_tensor[0,c].mean().cpu()) for c in range(3)],
                             [float(inp_tensor[0,c].std().cpu()) for c in range(3)])
        except Exception as e:
            st.error("Preprocessing failed.")
            st.exception(e)
            st.stop()

        # inference
        try:
            input_device = inp_tensor.to(DEVICE)
            with torch.no_grad():
                logits = model(input_device)  # B,C,H,W
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # C,H,W
            if probs.shape[0] == 1:
                prob_map = probs[0]
            else:
                prob_map = probs[1]  # class 1 prob map
            st.sidebar.write(f"prob_map stats: min {float(prob_map.min()):.4f}, max {float(prob_map.max()):.4f}, mean {float(prob_map.mean()):.4f}")
        except Exception as e:
            st.error("Inference failed.")
            st.exception(e)
            st.stop()

        # diagnostics
        if show_prob:
            fig, axs = plt.subplots(1,2, figsize=(10,3))
            axs[0].imshow(prob_map, cmap='inferno')
            axs[0].set_title("Class-1 probability")
            axs[0].axis('off')
            axs[1].hist(prob_map.ravel(), bins=80)
            axs[1].set_title("Probability histogram")
            st.pyplot(fig)

        # threshold and cleanup
        bin_mask = (prob_map >= threshold).astype(np.uint8)
        clean_mask = bin_mask.copy()
        if apply_cleanup and clean_mask.sum() > 0:
            if HAS_SCIPY:
                labeled, ncomp = ndi.label(clean_mask)
                if ncomp > 0:
                    counts = np.bincount(labeled.ravel())
                    remove = np.where(counts < min_area)[0]
                    remove = remove[remove != 0]
                    if remove.size > 0:
                        clean_mask[np.isin(labeled, remove)] = 0
            elif HAS_CV2:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask.astype(np.uint8), connectivity=8)
                for lbl in range(1, num_labels):
                    area = stats[lbl, cv2.CC_STAT_AREA]
                    if area < min_area:
                        clean_mask[labels == lbl] = 0
            else:
                # morphological opening to reduce dots
                try:
                    kernel = np.ones((3,3), dtype=np.uint8)
                    clean_mask = cv2.morphologyEx(clean_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                except Exception:
                    # fallback majority filter
                    pad = np.pad(clean_mask, 1, mode='constant', constant_values=0)
                    filt = np.zeros_like(clean_mask)
                    for i in range(clean_mask.shape[0]):
                        for j in range(clean_mask.shape[1]):
                            window = pad[i:i+3, j:j+3]
                            filt[i,j] = 1 if window.sum() >= 2 else 0
                    clean_mask = filt

        # visuals
        mask_before = mask_to_uint8(bin_mask)
        mask_after  = mask_to_uint8(clean_mask)
        overlay_after = colored_overlay(pil_img, clean_mask, alpha=overlay_alpha)

        cols = st.columns([1,1,1])
        with cols[0]:
            st.subheader("Input (resized)")
            st.image(pil_img, use_column_width=True)
        with cols[1]:
            st.subheader("Predicted Mask (raw threshold)")
            st.image(mask_before, use_column_width=True, clamp=True)
        with cols[2]:
            st.subheader("Mask (cleaned) & Overlay")
            st.image(mask_after, use_column_width=True, clamp=True)
            st.image(overlay_after, use_column_width=True)

        st.markdown("---")
        st.metric("Model mean confidence", f"{float(prob_map.mean()*100):.2f}%")
        st.sidebar.write("Cleaned pixels count:", int(clean_mask.sum()))

        # downloads: PNG mask
        png_buf = io.BytesIO()
        Image.fromarray(mask_after).save(png_buf, format='PNG')
        png_buf.seek(0)
        st.download_button("Download cleaned mask (PNG)", data=png_buf.getvalue(), file_name="pred_mask_clean.png", mime="image/png")

        # GeoTIFF export: safe nodata handling
        if tif_profile is not None and rasterio is not None:
            try:
                out_mask = (clean_mask.astype(np.uint8) * 255)
                orig_h = tif_profile.get("height", None)
                orig_w = tif_profile.get("width", None)
                if orig_h and orig_w and (orig_h != out_mask.shape[0] or orig_w != out_mask.shape[1]):
                    pil_mask = Image.fromarray(out_mask)
                    pil_mask = pil_mask.resize((orig_w, orig_h), resample=Image.NEAREST)
                    out_arr = np.array(pil_mask).astype(np.uint8)
                else:
                    out_arr = out_mask.astype(np.uint8)
                out_profile = tif_profile.copy()
                # remove nodata that isn't compatible with uint8
                out_profile.pop("nodata", None)
                out_profile.update({"dtype": "uint8", "count": 1, "compress": "lzw"})
                tmp_path = "pred_mask_georef.tif"
                with rasterio.open(tmp_path, "w", **out_profile) as dst:
                    dst.write(out_arr[np.newaxis, :, :])
                with open(tmp_path, "rb") as f:
                    st.download_button("Download cleaned mask (GeoTIFF)", data=f.read(), file_name="pred_mask_georef.tif", mime="image/tiff")
            except Exception as e:
                st.warning("GeoTIFF export failed (see details).")
                st.exception(e)
