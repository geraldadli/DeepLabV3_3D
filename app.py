# app.py
# Robust Streamlit app for Sentinel-2 tile inference with DinoV2 backbone
# - robust GeoTIFF reading (rasterio / tifffile fallback)
# - preprocessing matched to HF AutoImageProcessor when available
# - percentile stretch for satellite data
# - threshold + small-object cleanup controls
# - overlay blending (alpha) instead of direct replacement
# Paste this file into your repo and run with `streamlit run app.py`.

import io
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn

# Optional libs for robust TIFF handling / postprocessing
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
    import scipy.ndimage as ndi
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# transformers (DINOv2 model + processor)
try:
    from transformers import Dinov2Model, AutoImageProcessor
    HF_AVAILABLE = True
except Exception:
    Dinov2Model = None
    AutoImageProcessor = None
    HF_AVAILABLE = False

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "output/model_epoch6.pth"     # adjust if different
BACKBONE = "facebook/dinov2-small"
IMAGE_SIZE = 128                           # model input spatial size
NUM_CLASSES = 2

st.set_page_config(layout="wide", page_title="AgriVision AI - DinoV2 Seg")
st.title("🌾 AgriVision AI — Sentinel-2 crop segmentation (DinoV2)")
st.markdown("Upload a GeoTIFF (.tif/.tiff) or PNG/JPG. Use sidebar controls to adjust preprocessing and postprocessing.")

# ---------------- UI controls ----------------
st.sidebar.header("Preprocessing")
percentile_stretch = st.sidebar.checkbox("Apply percentile stretch (recommended for Sentinel-2)", value=True)
p_low = st.sidebar.slider("Percentile low", 0, 10, 2)
p_high = st.sidebar.slider("Percentile high", 90, 99, 98)
norm_mode = st.sidebar.selectbox("Normalization", options=["auto (HF processor if available)", "imagenet", "center-0.5"], index=0)

st.sidebar.header("Postprocessing")
show_prob = st.sidebar.checkbox("Show probability heatmap & histogram", value=True)
threshold = st.sidebar.slider("Probability threshold for class1", 0.01, 0.99, 0.5, 0.01)
min_area = st.sidebar.slider("Remove small objects < X px", 0, 5000, 50, 10)
apply_cleanup = st.sidebar.checkbox("Apply small-object removal", value=True)
overlay_alpha = st.sidebar.slider("Overlay alpha", 0.0, 1.0, 0.5)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("Run inference")

# ---------------- Model definition ----------------
class SimpleDinov2Seg(nn.Module):
    def __init__(self, backbone_name=BACKBONE, num_classes=NUM_CLASSES):
        super().__init__()
        if Dinov2Model is None:
            raise RuntimeError("transformers.Dinov2Model is not available in the environment.")
        self.backbone = Dinov2Model.from_pretrained(backbone_name)
        hidden_dim = self.backbone.config.hidden_size
        patch_size = getattr(self.backbone.config, "patch_size", 16)
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim//4, hidden_dim//8, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(hidden_dim//8, num_classes, kernel_size=1)

    def forward(self, x):
        # x: B,C,H,W
        out = self.backbone(pixel_values=x)["last_hidden_state"]  # B, tokens, hidden
        seq_len = out.shape[1]
        expected_seq = (x.shape[2] // self.patch_size) * (x.shape[3] // self.patch_size)
        # some HF backbones add cls token -> drop it
        if seq_len == expected_seq + 1:
            out = out[:, 1:, :]
        # reshape to spatial
        Hpatch = x.shape[2] // self.patch_size
        Wpatch = x.shape[3] // self.patch_size
        out = out.permute(0, 2, 1).contiguous().view(x.shape[0], self.hidden_dim, Hpatch, Wpatch)
        out = self.conv1(out)
        out = self.upsample(out)
        out = self.final_conv(out)
        out = nn.functional.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return out

# ---------------- Load model & processor (cached) ----------------
@st.cache_resource
def load_model_and_processor(model_path=MODEL_PATH, backbone=BACKBONE):
    processor = None
    if HF_AVAILABLE and AutoImageProcessor is not None:
        try:
            processor = AutoImageProcessor.from_pretrained(backbone)
        except Exception:
            processor = None

    model = SimpleDinov2Seg(backbone_name=backbone, num_classes=NUM_CLASSES)
    ckpt = torch.load(model_path, map_location="cpu")
    # support different checkpoint layouts
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt
    # load tolerant
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model, processor

try:
    model, processor = load_model_and_processor()
    st.sidebar.success("Model & processor loaded")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

# ---------------- Robust TIFF/JPEG reader ----------------
def read_image_from_upload(uploaded_file, image_size=IMAGE_SIZE, percentile=True, p_low=2, p_high=98):
    """
    Returns: pil_rgb_resized (PIL.Image), tif_profile_or_None, debug_info (dict)
    - Uses rasterio -> tifffile -> PIL fallback.
    - Applies percentile stretch if requested.
    """
    raw = uploaded_file.read()
    debug = {}
    filename = getattr(uploaded_file, "name", "").lower()

    # If TIFF, prefer rasterio/tifffile
    if filename.endswith((".tif", ".tiff")):
        if rasterio is not None and MemoryFile is not None:
            try:
                with MemoryFile(raw) as mem:
                    with mem.open() as src:
                        arr = src.read(out_dtype='float32')  # C,H,W
                        profile = src.profile.copy()
                        debug['reader'] = 'rasterio'
                        debug['orig_shape'] = arr.shape
                        # ensure 3 channels
                        if arr.shape[0] >= 3:
                            rgb = arr[:3]
                        else:
                            reps = int(np.ceil(3.0 / max(1, arr.shape[0])))
                            rgb = np.tile(arr, (reps, 1, 1))[:3]
                        if percentile:
                            lo = np.nanpercentile(rgb, p_low)
                            hi = np.nanpercentile(rgb, p_high)
                            rgb = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)
                        else:
                            mn = rgb.min(); mx = rgb.max()
                            rgb = (rgb - mn) / (mx - mn + 1e-9)
                        img = np.moveaxis(rgb, 0, -1)  # H,W,3
                        pil = Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8)).convert("RGB")
                        pil = pil.resize((image_size, image_size), resample=Image.BILINEAR)
                        return pil, profile, debug
            except Exception as e:
                debug['rasterio_error'] = str(e)
        # tifffile fallback
        if tifffile is not None:
            try:
                arr = tifffile.imread(io.BytesIO(raw))
                arr = np.asarray(arr)
                debug['reader'] = 'tifffile'
                debug['orig_shape'] = arr.shape
                # normalize arr to H,W,C
                if arr.ndim == 2:
                    arr = np.stack([arr,arr,arr], axis=-1)
                elif arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[0] != arr.shape[2]:
                    # (C,H,W) -> (H,W,C)
                    arr = np.moveaxis(arr, 0, -1)
                if arr.shape[2] >= 3:
                    rgb = arr[:,:,:3].astype('float32')
                else:
                    rgb = np.repeat(arr, 3, axis=2)[:,:,:3].astype('float32')
                if percentile:
                    lo = np.nanpercentile(rgb, p_low)
                    hi = np.nanpercentile(rgb, p_high)
                    rgb = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)
                else:
                    mn = rgb.min(); mx = rgb.max()
                    rgb = (rgb - mn) / (mx - mn + 1e-9)
                pil = Image.fromarray((np.clip(rgb,0,1)*255).astype(np.uint8)).convert("RGB")
                pil = pil.resize((image_size, image_size), resample=Image.BILINEAR)
                return pil, None, debug
            except Exception as e:
                debug['tifffile_error'] = str(e)
        # no TIFF reader succeeded
        raise RuntimeError(f"Unable to read TIFF. Install rasterio or tifffile. Debug: {debug}")
    else:
        # try PIL for other formats
        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
            pil = pil.resize((image_size, image_size), resample=Image.BILINEAR)
            return pil, None, {'reader':'PIL'}
        except Exception as e:
            raise RuntimeError(f"PIL could not read file: {e}")

# ---------------- Preprocessing that matches HF processor when possible ----------------
def pil_to_tensor(pil_img, processor=None, norm_mode="auto"):
    """
    Convert PIL image to 1xCxHxW torch.float32 tensor (CPU).
    If processor (AutoImageProcessor) available and norm_mode == 'auto', use its mean/std.
    Returns tensor and a description of the normalization used.
    """
    arr = np.array(pil_img).astype('float32') / 255.0  # H,W,3
    tensor = torch.tensor(arr).permute(2,0,1).unsqueeze(0).float()  # 1,C,H,W

    if norm_mode == "auto" and processor is not None and hasattr(processor, "image_mean") and hasattr(processor, "image_std"):
        mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1,3,1,1)
        std  = torch.tensor(processor.image_std, dtype=torch.float32).view(1,3,1,1)
        tensor = (tensor - mean) / (std + 1e-9)
        return tensor, ("hf_processor", processor.image_mean, processor.image_std)

    if norm_mode == "imagenet":
        mean = torch.tensor([0.485,0.456,0.406], dtype=torch.float32).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225], dtype=torch.float32).view(1,3,1,1)
        tensor = (tensor - mean) / (std + 1e-9)
        return tensor, ("imagenet", [0.485,0.456,0.406], [0.229,0.224,0.225])

    # default centered 0.5
    mean = torch.tensor([0.5,0.5,0.5], dtype=torch.float32).view(1,3,1,1)
    std  = torch.tensor([0.5,0.5,0.5], dtype=torch.float32).view(1,3,1,1)
    tensor = (tensor - mean) / (std + 1e-9)
    return tensor, ("center05", [0.5,0.5,0.5], [0.5,0.5,0.5])

# ---------------- Small-object removal ----------------
def remove_small_objects(mask, min_size):
    """
    mask: 2D uint8/boolean numpy array (0/1)
    returns cleaned mask (0/1)
    """
    if min_size <= 0:
        return mask
    if not mask.any():
        return mask
    if HAS_SCIPY:
        labeled, ncomp = ndi.label(mask)
        if ncomp == 0:
            return mask
        counts = np.bincount(labeled.ravel())
        remove = np.where(counts < min_size)[0]
        remove = remove[remove != 0]
        if remove.size > 0:
            mask[np.isin(labeled, remove)] = 0
        return mask
    else:
        # fallback: simple area filter by connected components via iterative flood fill (slow but works)
        h,w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        out = np.zeros_like(mask, dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                if mask[i,j] and not visited[i,j]:
                    # BFS flood fill
                    stack = [(i,j)]
                    comp = []
                    visited[i,j] = True
                    while stack:
                        x,y = stack.pop()
                        comp.append((x,y))
                        for dx in (-1,0,1):
                            for dy in (-1,0,1):
                                nx,ny = x+dx, y+dy
                                if 0 <= nx < h and 0 <= ny < w and not visited[nx,ny] and mask[nx,ny]:
                                    visited[nx,ny] = True
                                    stack.append((nx,ny))
                    if len(comp) >= min_size:
                        for (x,y) in comp:
                            out[x,y] = 1
        return out

# ---------------- Visualization helpers ----------------
def mask_to_png(mask):
    # mask is 2D 0/1
    return (mask.astype(np.uint8) * 255)

def blend_overlay(pil_img, mask, alpha=0.5, color=(255,0,0)):
    base = np.array(pil_img).astype(np.float32)
    overlay = np.zeros_like(base, dtype=np.uint8)
    overlay[mask==1] = color
    blended = (base * (1.0 - alpha) + overlay.astype(np.float32) * alpha).clip(0,255).astype(np.uint8)
    return blended

# ---------------- Upload UI ----------------
uploaded = st.file_uploader("Upload Sentinel-2 tile (.tif/.tiff) or JPG/PNG", type=["tif","tiff","jpg","jpeg","png"])
if uploaded is None:
    st.info("Upload a file to start. Use sidebar controls to tune preprocessing and postprocessing.")
else:
    st.markdown(f"**File:** `{getattr(uploaded,'name','')}` — size: {uploaded.size/1024:.1f} KB")

    if run_btn:
        # read image
        try:
            pil_img, tif_profile, debug = read_image_from_upload(
                uploaded,
                image_size=IMAGE_SIZE,
                percentile=percentile_stretch,
                p_low=p_low,
                p_high=p_high
            )
        except Exception as e:
            st.error("Failed to read uploaded file. See error below.")
            st.exception(e)
            st.stop()

        st.write("Reader debug:", debug)

        # preprocess -> tensor
        try:
            norm_choice = "auto" if norm_mode.startswith("auto") else ("imagenet" if norm_mode=="imagenet" else "center05")
            inp_tensor, used_norm = pil_to_tensor(pil_img, processor=processor, norm_mode=norm_choice)
            st.sidebar.write("Normalization used:", used_norm)
            st.sidebar.write("Input tensor mean per channel (first batch):",
                             [float(inp_tensor[0,c].mean().cpu()) for c in range(3)])
        except Exception as e:
            st.error("Preprocessing failed")
            st.exception(e)
            st.stop()

        # model inference
        try:
            inp = inp_tensor.to(DEVICE)
            with torch.no_grad():
                logits = model(inp)  # B,C,H,W
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # C,H,W
            if probs.shape[0] == 1:
                prob_map = probs[0]
            else:
                # assume binary segmentation: class 1 probability
                prob_map = probs[1]
            st.sidebar.write(f"prob_map stats - min:{prob_map.min():.4f} max:{prob_map.max():.4f} mean:{prob_map.mean():.4f}")
        except Exception as e:
            st.error("Inference failed")
            st.exception(e)
            st.stop()

        # show prob map & histogram
        if show_prob:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1,2, figsize=(10,3))
            axs[0].imshow(prob_map, cmap='inferno')
            axs[0].set_title("class-1 probability")
            axs[0].axis('off')
            axs[1].hist(prob_map.ravel(), bins=80)
            axs[1].set_title("probability histogram")
            st.pyplot(fig)

        # threshold -> binary mask
        bin_mask = (prob_map >= threshold).astype(np.uint8)

        # cleanup
        if apply_cleanup and bin_mask.sum() > 0:
            clean_mask = remove_small_objects(bin_mask.copy(), min_area)
        else:
            clean_mask = bin_mask

        # visuals
        mask_vis = mask_to_png(clean_mask)
        overlay_img = blend_overlay(pil_img, clean_mask, alpha=overlay_alpha)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Input (resized)")
            st.image(pil_img, use_column_width=True)
        with c2:
            st.subheader("Predicted Mask")
            st.image(mask_vis, use_column_width=True, clamp=True)
        with c3:
            st.subheader("Overlay (red = class1)")
            st.image(overlay_img, use_column_width=True)

        st.markdown("---")
        st.metric("Mean model confidence (class1)", f"{float(prob_map.mean()*100):.2f}%")
        st.sidebar.write("Cleaned pixels count:", int(clean_mask.sum()))

        # Download mask PNG
        png_buf = io.BytesIO()
        Image.fromarray(mask_vis).save(png_buf, format="PNG")
        png_buf.seek(0)
        st.download_button("Download mask (PNG)", data=png_buf.getvalue(), file_name="pred_mask.png", mime="image/png")

        # GeoTIFF export if original profile & rasterio available
        if tif_profile is not None and rasterio is not None:
            try:
                out_arr = mask_vis.astype(np.uint8)
                # if original size differs, resize back
                orig_h = tif_profile.get("height", None)
                orig_w = tif_profile.get("width", None)
                if orig_h and orig_w and (orig_h != out_arr.shape[0] or orig_w != out_arr.shape[1]):
                    pil_mask = Image.fromarray(out_arr)
                    pil_mask = pil_mask.resize((orig_w, orig_h), resample=Image.NEAREST)
                    out_arr_full = np.array(pil_mask).astype(np.uint8)
                else:
                    out_arr_full = out_arr.astype(np.uint8)
                out_profile = tif_profile.copy()
                # remove incompatible nodata (e.g., negative) and ensure uint8 dtype
                out_profile.pop("nodata", None)
                out_profile.update({"dtype": "uint8", "count": 1, "compress": "lzw"})
                tmp_path = "pred_mask_georef.tif"
                with rasterio.open(tmp_path, "w", **out_profile) as dst:
                    dst.write(out_arr_full[np.newaxis, :, :])
                with open(tmp_path, "rb") as f:
                    st.download_button("Download mask (GeoTIFF)", data=f.read(), file_name="pred_mask_georef.tif", mime="image/tiff")
            except Exception as e:
                st.warning("GeoTIFF export failed; see details.")
                st.exception(e)
