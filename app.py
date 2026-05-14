"""
Visual Product Search Engine — Streamlit Demo
VR Course Final Project

Run on Kaggle (in a code cell):
    !pip install -q streamlit open_clip_torch hnswlib pyngrok transformers accelerate
    !pkill -f streamlit; sleep 1
    import threading, subprocess
    from pyngrok import ngrok
    def run():
        subprocess.run(["streamlit", "run", "/kaggle/working/app.py",
                        "--server.port", "8501", "--server.headless", "true"])
    threading.Thread(target=run, daemon=True).start()
    import time; time.sleep(8)
    tunnel = ngrok.connect(8501)
    print("App URL:", tunnel.public_url)

Run locally:
    pip install streamlit pillow numpy pandas open_clip_torch hnswlib transformers accelerate torch
    streamlit run app.py
"""

import json, pickle, logging, time
import torch
import torch.nn.functional as F
import hnswlib
import open_clip
import numpy as np
import streamlit as st

from pathlib import Path
from PIL import Image, ImageDraw
from transformers import (
    BlipProcessor, BlipForImageTextRetrieval,
    AutoImageProcessor, AutoModelForObjectDetection,
)

logging.getLogger("root").setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

ON_KAGGLE = Path("/kaggle").exists()

if ON_KAGGLE:
    CHKPT_DIR     = Path("/kaggle/input/datasets/rrickyroger/checkpoints")
    CAPTIONS_PATH = Path("/kaggle/input/datasets/rrickyroger/captions/captions.json")
    IMG_ROOT      = Path("/kaggle/input/datasets/abhinavkishan123/deepfashion-inshop-dataset/Dataset")
else:
    _HERE         = Path(__file__).parent
    CHKPT_DIR     = _HERE / "checkpoints"
    CAPTIONS_PATH = _HERE / "captions" / "captions.json"
    IMG_ROOT      = _HERE / "data" / "Dataset"

HNSW_INDEX    = CHKPT_DIR / "hnsw_index.bin"
METADATA_FILE = CHKPT_DIR / "index_metadata.pkl"
CLIP_CKPT     = CHKPT_DIR / "clip_finetuned.pt"

# Alpha-specific HNSW index files (alpha controls cosine vs ITM blend weight)
# 1.0 = pure cosine, 0.6 = balanced, 0.3 = ITM-heavy
HNSW_INDEX_BY_ALPHA = {
    1.0: CHKPT_DIR / "hnsw_1.0_alpha10.bin",
    0.6: CHKPT_DIR / "hnsw_index.bin",      # original index, used as fallback for 0.6
    0.3: CHKPT_DIR / "hnsw_0.6_alpha3.bin",
}

NMS_IOU_THRESHOLD = 0.80  # suppress boxes overlapping more than this

CONF_THRESHOLD = 0.25
MIN_BBOX_RATIO = 0.05
PADDING        = 0.08

FASHION_CATEGORIES = {
    'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan',
    'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit',
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG + CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Visual Product Search", page_icon="🔍",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
section[data-testid="stSidebar"]{background:#0f0f0f;}
section[data-testid="stSidebar"] *{color:#e0e0e0!important;}
.sim-badge{display:inline-block;background:#6c63ff22;border:1px solid #6c63ff;
  border-radius:20px;padding:2px 10px;font-size:12px;font-family:'Space Mono',monospace;
  color:#a89cff;margin-top:6px;}
.rank-badge{display:inline-block;background:#ff636322;border:1px solid #ff6363;
  border-radius:20px;padding:2px 10px;font-size:12px;font-family:'Space Mono',monospace;
  color:#ff9999;margin-top:4px;}
.metric-box{background:#111827;border-radius:10px;padding:14px 18px;
  border-left:4px solid #6c63ff;margin-bottom:10px;}
.metric-label{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px;}
.metric-value{font-size:22px;font-weight:600;color:#e0e0e0;font-family:'Space Mono',monospace;}
.step-chip{display:inline-block;background:#6c63ff;color:white;border-radius:999px;
  padding:3px 14px;font-size:12px;font-weight:600;margin-bottom:10px;}
.section-header{font-size:1rem;font-weight:600;color:#e0e0e0;
  border-bottom:2px solid #6c63ff;padding-bottom:6px;margin-bottom:12px;}
.pre-rank-label{color:#63b3ff;font-size:0.82rem;font-weight:600;
  text-transform:uppercase;letter-spacing:1px;}
.post-rank-label{color:#ff9999;font-size:0.82rem;font-weight:600;
  text-transform:uppercase;letter-spacing:1px;}
.fullbody-chip{display:inline-block;background:#22c55e22;border:1px solid #22c55e;
  border-radius:20px;padding:2px 10px;font-size:11px;color:#86efac;
  font-family:'Space Mono',monospace;margin-left:8px;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING  (cached — loads once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading CLIP…")
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None)
    model.load_state_dict(torch.load(CLIP_CKPT, map_location=DEVICE, weights_only=False))
    model.eval().to(DEVICE)
    return model, preprocess

@st.cache_resource(show_spinner="Loading YOLOS…")
def load_yolos():
    processor = AutoImageProcessor.from_pretrained("valentinafevu/yolos-fashionpedia")
    model = AutoModelForObjectDetection.from_pretrained(
        "valentinafevu/yolos-fashionpedia").to(DEVICE)
    model.eval()
    return processor, model

@st.cache_resource(show_spinner="Loading BLIP ITM…")
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model = BlipForImageTextRetrieval.from_pretrained(
        "Salesforce/blip-itm-base-coco", torch_dtype=dtype).to(DEVICE)
    model.eval()
    return processor, model

@st.cache_resource(show_spinner="Loading HNSW index…")
def load_index(alpha: float = 1.0):
    with open(METADATA_FILE, "rb") as f:
        meta = pickle.load(f)
    alpha_path = HNSW_INDEX_BY_ALPHA.get(alpha, HNSW_INDEX)
    if alpha_path.exists():
        index_path = alpha_path
        st.sidebar.success(f"Loaded alpha index: `{alpha_path.name}`")
    else:
        index_path = HNSW_INDEX
        st.sidebar.warning(f"⚠️ `{alpha_path.name}` not found — fell back to `{HNSW_INDEX.name}`")
    idx = hnswlib.Index(space="cosine", dim=512)
    idx.load_index(str(index_path))
    idx.set_ef(50)
    return idx, meta

@st.cache_resource(show_spinner="Loading captions…")
def load_captions():
    with open(CAPTIONS_PATH) as f:
        raw = json.load(f)
    norm = {}
    for k, v in raw.items():
        if "img/img/" in k:
            norm["img/img/" + k.split("img/img/")[-1]] = v
    return norm

clip_model,   clip_preprocess = load_clip()
yolo_proc,    yolo_model      = load_yolos()
blip_proc,    blip_model      = load_blip()
captions_map                  = load_captions()
# Index is loaded dynamically based on selected alpha (see sidebar)

# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def run_yolo_detection(image: Image.Image) -> list[dict]:
    img_w, img_h = image.size
    inputs = yolo_proc(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = yolo_model(**inputs)

    target_sizes = torch.tensor([[img_h, img_w]]).to(DEVICE)
    results = yolo_proc.post_process_object_detection(
        outputs, threshold=CONF_THRESHOLD, target_sizes=target_sizes
    )[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = yolo_model.config.id2label.get(label.item(), "")
        if label_name not in FASHION_CATEGORIES:
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        if ((x2-x1)/img_w) < MIN_BBOX_RATIO or ((y2-y1)/img_h) < MIN_BBOX_RATIO:
            continue
        pw = int((x2-x1) * PADDING); ph = int((y2-y1) * PADDING)
        cx1 = max(0, x1-pw); cy1 = max(0, y1-ph)
        cx2 = min(img_w, x2+pw); cy2 = min(img_h, y2+ph)
        detections.append({
            "label": label_name,
            "score": round(score.item(), 3),
            "bbox":  (x1, y1, x2, y2),
            "crop":  image.crop((cx1, cy1, cx2, cy2)),
        })

    detections.sort(key=lambda x: x["score"], reverse=True)

    # ── Non-Maximum Suppression ────────────────────────────────────────────────
    # Remove boxes that overlap more than NMS_OVERLAP_THRESHOLD with a higher-conf box
    # We use Intersection over Minimum area (IoM) because often one box is inside another
    def _overlap(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2-ax1) * (ay2-ay1)
        area_b = (bx2-bx1) * (by2-by1)
        # Using intersection over minimum area to suppress smaller nested boxes
        return inter / min(area_a, area_b)

    kept = []
    for det in detections:
        suppressed = any(
            _overlap(det["bbox"], k["bbox"]) > NMS_IOU_THRESHOLD
            for k in kept
        )
        if not suppressed:
            kept.append(det)
    detections = kept
    # ── end NMS ───────────────────────────────────────────────────────────────

    if not detections:
        detections = [{
            "label": "Full image (no garment detected)",
            "score": 1.0,
            "bbox":  (0, 0, img_w, img_h),
            "crop":  image,
        }]

    return detections


def draw_detections(image: Image.Image, detections: list[dict],
                    highlight_idx: int | None = None) -> Image.Image:
    import colorsys
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    n = len(detections)
    for i, det in enumerate(detections):
        if det["label"] == "Full image (no garment detected)":
            continue
        hue = i / max(n, 1)
        r, g, b = [int(c*255) for c in colorsys.hsv_to_rgb(hue, 0.85, 0.95)]
        colour = f"#{r:02x}{g:02x}{b:02x}"
        lw = 4 if i == highlight_idx else 2
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=lw)
        tag = f"{i+1}. {det['label']}"
        draw.rectangle([x1, y1-18, x1 + len(tag)*7+6, y1], fill=colour)
        draw.text((x1+3, y1-16), tag, fill="black")
    return vis


def run_clip_embed(crop: Image.Image) -> np.ndarray:
    with torch.no_grad():
        tensor = clip_preprocess(crop).unsqueeze(0).to(DEVICE)
        emb = F.normalize(clip_model.encode_image(tensor), dim=-1)
    vec = emb.cpu().float().numpy()
    vec = vec / np.linalg.norm(vec)
    return vec


def run_hnsw_retrieval(query_vec: np.ndarray, top_k: int) -> list[dict]:
    labels, distances = hnsw_index.knn_query(query_vec, k=top_k)
    results = []
    for idx, dist in zip(labels[0], distances[0]):
        meta = gallery_meta[idx]
        results.append({
            **meta,
            "cosine_score": round(float(1 - dist), 4),
            "itm_score": None,
            "pre_rank": None,   # filled later
        })
    # store original CLIP rank
    for i, r in enumerate(results):
        r["pre_rank"] = i + 1
    return results


def run_blip_rerank(crop: Image.Image, candidates: list[dict], alpha: float = 1.0) -> list[dict]:
    """Returns a NEW list sorted by blended score; alpha controls cosine vs ITM weight.
    
    final_score = alpha * cosine_score + (1 - alpha) * itm_score
    alpha=1.0 → pure cosine order; alpha=0.0 → pure ITM order
    """
    scored = []
    for c in candidates:
        img_name = c["image_name"]
        key = "img/img/" + img_name.split("img/img/")[-1] if "img/img/" in img_name else img_name
        caption = captions_map.get(key, "fashion clothing")
        inputs = blip_proc(
            images=crop, text=caption, return_tensors="pt", padding=True
        ).to(DEVICE)
        with torch.no_grad():
            score = torch.softmax(blip_model(**inputs).itm_score, dim=1)[0][1].item()
        blended = alpha * c["cosine_score"] + (1.0 - alpha) * score
        scored.append({**c, "itm_score": round(score, 4), "blended_score": round(blended, 4)})
    return sorted(scored, key=lambda x: x["blended_score"], reverse=True)


def load_gallery_image(image_name: str) -> Image.Image | None:
    path = IMG_ROOT / image_name
    if path.exists():
        return Image.open(path).convert("RGB")
    return None


def get_caption(image_name: str) -> str:
    key = "img/img/" + image_name.split("img/img/")[-1] if "img/img/" in image_name else image_name
    return captions_map.get(key, "fashion clothing")


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

defaults = {
    "stage":            "upload",
    "query_image":      None,
    "detections":       None,
    "selected_idx":     None,
    "cropped_image":    None,
    "pre_rerank":       None,   # list[dict] — CLIP order
    "post_rerank":      None,   # list[dict] — BLIP order
    "elapsed":          {},
    "full_body_mode":   False,  # True → skip YOLO, use whole image
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def reset():
    for k, v in defaults.items():
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Search Settings")
    st.markdown("---")
    top_k = st.slider("Top-K results", 5, 15, 10, 5)
    use_rerank = st.toggle("BLIP ITM re-ranking", value=True)

    st.markdown("---")
    st.markdown("**⚖️ Alpha (cosine ↔ ITM blend)**")
    alpha = st.radio(
        "Alpha",
        [1.0, 0.6, 0.3],
        index=1,
        format_func=lambda a: {
            1.0: "α = 1.0 — pure cosine similarity",
            0.6: "α = 0.6 — balanced blend",
            0.3: "α = 0.3 — ITM-heavy blend",
        }[a],
        help=(
            "Controls how cosine score and BLIP ITM score are combined during re-ranking.\n\n"
            "final = α × cosine + (1−α) × ITM\n\n"
            "Each alpha value loads a different HNSW index file."
        ),
    )
    # Load the index for the selected alpha (cached per alpha value)
    hnsw_index, gallery_meta = load_index(alpha)

    st.markdown("---")
    st.markdown("**🔲 Search Mode**")
    full_body_mode = st.toggle(
        "Full body search (skip YOLO crop)",
        value=st.session_state.full_body_mode,
        help="When ON, the entire uploaded image is used as the query instead of "
             "a YOLO-detected crop. Useful when you want to match the full outfit.",
    )
    st.session_state.full_body_mode = full_body_mode
    if full_body_mode:
        st.markdown(
            "<span class='fullbody-chip'>YOLO skipped</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Ablation config**")
    ablation = st.radio("Condition", [
        "A — Vision-only CLIP (no rerank)",
        "B — Frozen CLIP + BLIP-2",
        "C — Fine-tuned CLIP + BLIP-2",
    ], index=2)
    if ablation.startswith("A"):
        use_rerank = False

    st.markdown("---")
    st.markdown("**Model status**")
    yolo_status = "⬜ YOLOS (skipped)" if full_body_mode else "🟢 YOLOS"
    st.markdown(f"{yolo_status} &nbsp;&nbsp; `yolos-fashionpedia`")
    st.markdown(f"🟢 CLIP &nbsp;&nbsp;&nbsp; `ViT-B-32 (fine-tuned)`")
    st.markdown(f"🟢 BLIP &nbsp;&nbsp;&nbsp; `blip-itm-base-coco`")
    _alpha_path = HNSW_INDEX_BY_ALPHA.get(alpha, HNSW_INDEX)
    _loaded_path = _alpha_path if _alpha_path.exists() else HNSW_INDEX
    st.markdown(f"🟢 HNSW &nbsp;&nbsp; `{hnsw_index.get_current_count()} items` · `{_loaded_path.name}`")
    st.markdown(f"💻 Device: `{DEVICE}` &nbsp;|&nbsp; NMS IoU ≥ `{NMS_IOU_THRESHOLD}`")

    st.markdown("---")
    if st.button("🔄 Reset", use_container_width=True):
        reset(); st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER + BREADCRUMB
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<h1 style='font-family:DM Sans;font-weight:600;font-size:2rem;margin-bottom:0'>
    🔍 Visual Product Search Engine
</h1>
<p style='color:#888;font-size:0.9rem;margin-top:4px'>
    DeepFashion · YOLOS + CLIP (fine-tuned) + BLIP ITM + HNSW &nbsp;|&nbsp; VR Course Final Project
</p>
<hr style='border-color:#222;margin-bottom:1.5rem'>
""", unsafe_allow_html=True)

# Stage labels depend on mode
if st.session_state.full_body_mode:
    STAGE_LABELS = ["📤 Upload", "🏆 Results"]
    STAGE_IDX    = {"upload": 0, "results": 1}
    stage_count  = 2
else:
    STAGE_LABELS = ["📤 Upload", "🔎 Detect", "✂️ Confirm", "🏆 Results"]
    STAGE_IDX    = {"upload": 0, "detect": 1, "confirm": 2, "results": 3}
    stage_count  = 4

cur_stage = st.session_state.stage
# Map stage name → breadcrumb index safely
cur_idx = STAGE_IDX.get(cur_stage, 0)

for i, (col, label) in enumerate(zip(st.columns(stage_count), STAGE_LABELS)):
    with col:
        colour = "#6c63ff" if i == cur_idx else ("#aaa" if i < cur_idx else "#333")
        st.markdown(
            f"<div style='text-align:center;font-weight:{'600' if i==cur_idx else '400'};"
            f"color:{colour};border-bottom:2px solid {colour};padding-bottom:4px'>{label}</div>",
            unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HELPER — run search from a crop image
# ══════════════════════════════════════════════════════════════════════════════

def execute_search(crop: Image.Image):
    """Embed → retrieve → (optionally) rerank. Stores into session_state."""
    with st.spinner("CLIP embedding…"):
        t0 = time.time()
        q_vec = run_clip_embed(crop)
        st.session_state.elapsed["clip"] = round(time.time()-t0, 3)

    with st.spinner(f"HNSW retrieval — top {top_k}…"):
        t0 = time.time()
        pre = run_hnsw_retrieval(q_vec, top_k)
        st.session_state.elapsed["hnsw"] = round(time.time()-t0, 3)

    st.session_state.pre_rerank = pre

    if use_rerank:
        with st.spinner("BLIP ITM re-ranking…"):
            t0 = time.time()
            post = run_blip_rerank(crop, pre, alpha=alpha)
            st.session_state.elapsed["blip"] = round(time.time()-t0, 3)
        # add post_rank index
        for i, r in enumerate(post):
            r["post_rank"] = i + 1
        st.session_state.post_rerank = post
    else:
        st.session_state.post_rerank = None

    st.session_state.stage = "results"


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.stage == "upload":
    col_up, col_hint = st.columns([1, 1], gap="large")

    with col_up:
        mode_chip = (
            "<span class='fullbody-chip'>Full body mode ON</span>"
            if st.session_state.full_body_mode else ""
        )
        st.markdown(
            f'<div class="step-chip">Step 1 · Upload a clothing image{mode_chip}</div>',
            unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop image here",
                                    type=["jpg","jpeg","png","webp"],
                                    label_visibility="collapsed")
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.session_state.query_image = img
            st.image(img, caption="Uploaded image", use_container_width=True)

            if st.session_state.full_body_mode:
                # Skip YOLO entirely — go straight to search
                if st.button("🚀 Search with full image →", type="primary", use_container_width=True):
                    st.session_state.cropped_image = img
                    # Create a dummy detection for display purposes
                    w, h = img.size
                    st.session_state.detections = [{
                        "label": "Full image (full body mode)",
                        "score": 1.0,
                        "bbox": (0, 0, w, h),
                        "crop": img,
                    }]
                    st.session_state.selected_idx = 0
                    execute_search(img)
                    st.rerun()
            else:
                if st.button("🚀 Run YOLOS detection →", type="primary", use_container_width=True):
                    with st.spinner("Running YOLOS fashion detection…"):
                        t0 = time.time()
                        dets = run_yolo_detection(img)
                        st.session_state.elapsed["yolo"] = round(time.time()-t0, 3)
                        st.session_state.detections = dets
                        st.session_state.stage = "detect"
                    st.rerun()

    with col_hint:
        if st.session_state.full_body_mode:
            st.markdown("""
            <div style='background:#111827;border-radius:12px;padding:24px;border:1px solid #1e293b'>
            <h4 style='color:#e0e0e0;margin-top:0'>Pipeline (Full Body Mode)</h4>
            <ol style='color:#888;font-size:0.88rem;line-height:1.9'>
              <li><b style='color:#22c55e'>YOLO skipped</b> — full image used as query</li>
              <li><b style='color:#a89cff'>CLIP ViT-B/32</b> (fine-tuned) embeds the whole image</li>
              <li><b style='color:#a89cff'>HNSW</b> retrieves top-K nearest neighbours</li>
              <li><b style='color:#a89cff'>BLIP ITM</b> re-ranks candidates (if enabled)</li>
              <li>Results shown <b style='color:#e0e0e0'>pre- and post-reranking</b></li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#111827;border-radius:12px;padding:24px;border:1px solid #1e293b'>
            <h4 style='color:#e0e0e0;margin-top:0'>Pipeline</h4>
            <ol style='color:#888;font-size:0.88rem;line-height:1.9'>
              <li><b style='color:#63b3ff'>YOLOS</b> detects all fashion items in the image</li>
              <li>You <b style='color:#e0e0e0'>select</b> which detected garment to search for</li>
              <li><b style='color:#a89cff'>CLIP ViT-B/32</b> (fine-tuned) embeds the crop</li>
              <li><b style='color:#a89cff'>HNSW</b> retrieves top-K nearest neighbours</li>
              <li><b style='color:#a89cff'>BLIP ITM</b> re-ranks candidates by image-text match</li>
              <li>Results shown <b style='color:#e0e0e0'>pre- and post-reranking</b></li>
            </ol>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — DETECTION SELECTION  (only in normal mode)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.stage == "detect":
    dets = st.session_state.detections
    st.markdown('<div class="step-chip">Step 2 · Select the garment to search for</div>',
                unsafe_allow_html=True)

    col_ov, col_sel = st.columns([1, 2], gap="large")

    with col_ov:
        st.markdown("**YOLOS detections**")
        hi = st.session_state.selected_idx
        overview = draw_detections(st.session_state.query_image, dets, highlight_idx=hi)
        st.image(overview, use_container_width=True)
        st.caption(f"Found **{len(dets)}** fashion item(s) &nbsp;|&nbsp; "
                   f"YOLOS: `{st.session_state.elapsed.get('yolo','–')} s`")

    with col_sel:
        st.markdown(f"**{len(dets)} garment(s) detected — click one to select:**")

        cols_per_row = min(3, len(dets))
        for row_start in range(0, len(dets), cols_per_row):
            row_dets = dets[row_start: row_start + cols_per_row]
            row_cols = st.columns(cols_per_row, gap="medium")
            for col, (det, abs_idx) in zip(
                row_cols,
                [(d, row_start+i) for i, d in enumerate(row_dets)]
            ):
                with col:
                    is_sel = st.session_state.selected_idx == abs_idx
                    border = "3px solid #6c63ff" if is_sel else "2px solid #2a2a4a"
                    st.image(det["crop"], use_container_width=True)
                    st.markdown(
                        f"<div style='text-align:center;border:{border};"
                        f"border-radius:10px;padding:8px;background:{'#1a1a35' if is_sel else '#111827'}'>"
                        f"<div style='font-size:13px;font-weight:600;color:#e0e0e0'>"
                        f"{abs_idx+1}. {det['label']}</div>"
                        f"<div style='display:inline-block;background:#6c63ff22;border:1px solid #6c63ff;"
                        f"border-radius:20px;padding:2px 10px;font-size:11px;color:#a89cff;"
                        f"font-family:Space Mono,monospace'>conf {det['score']:.3f}</div>"
                        f"</div>",
                        unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    btn_label = "✅ Selected" if is_sel else "Select"
                    btn_type  = "primary" if is_sel else "secondary"
                    if st.button(btn_label, key=f"det_{abs_idx}",
                                 use_container_width=True, type=btn_type):
                        st.session_state.selected_idx = abs_idx
                        st.rerun()

    st.markdown("---")
    if st.session_state.selected_idx is not None:
        sel = dets[st.session_state.selected_idx]
        st.success(f"**{sel['label']}** selected (conf {sel['score']:.3f})")
        col_go, col_back, _ = st.columns([1, 1, 4])
        with col_go:
            if st.button("✂️ Confirm & review crop →", type="primary", use_container_width=True):
                st.session_state.cropped_image = sel["crop"]
                st.session_state.stage = "confirm"
                st.rerun()
        with col_back:
            if st.button("⬅️ Re-upload", use_container_width=True):
                reset(); st.rerun()
    else:
        st.info("👆 Click a detected garment above to continue.")

# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — CROP CONFIRM  (only in normal mode)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.stage == "confirm":
    sel = st.session_state.detections[st.session_state.selected_idx]
    st.markdown(f'<div class="step-chip">Step 3 · Confirm crop — {sel["label"]}</div>',
                unsafe_allow_html=True)

    col_orig, col_crop = st.columns(2, gap="large")
    with col_orig:
        st.markdown("**Original with selected region**")
        vis = draw_detections(st.session_state.query_image,
                              st.session_state.detections,
                              highlight_idx=st.session_state.selected_idx)
        st.image(vis, use_container_width=True)
        x1,y1,x2,y2 = sel["bbox"]
        st.caption(f"Label: **{sel['label']}** · conf `{sel['score']:.3f}` · "
                   f"box ({x1},{y1})→({x2},{y2})")

    with col_crop:
        st.markdown("**Cropped region (padded) — query input to CLIP**")
        st.image(st.session_state.cropped_image, use_container_width=True)

    st.markdown("---")
    col_confirm, col_change, col_redetect, _ = st.columns([1.2, 1, 1, 3])

    with col_confirm:
        if st.button("✅ Confirm & search", type="primary", use_container_width=True):
            execute_search(st.session_state.cropped_image)
            st.rerun()

    with col_change:
        if st.button("🔎 Change selection", use_container_width=True):
            st.session_state.stage = "detect"; st.rerun()

    with col_redetect:
        if st.button("🔄 Re-detect", use_container_width=True):
            with st.spinner("Re-running YOLOS…"):
                t0 = time.time()
                dets = run_yolo_detection(st.session_state.query_image)
                st.session_state.elapsed["yolo"] = round(time.time()-t0, 3)
                st.session_state.detections   = dets
                st.session_state.selected_idx = None
            st.session_state.stage = "detect"; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 — RESULTS  (pre- AND post-reranking side by side)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.stage == "results":
    pre_results  = st.session_state.pre_rerank
    post_results = st.session_state.post_rerank
    sel = st.session_state.detections[st.session_state.selected_idx]

    is_full_body = st.session_state.full_body_mode

    # ── top bar ────────────────────────────────────────────────────────────────
    col_q, col_info = st.columns([1, 3], gap="large")

    with col_q:
        label_text = "Full image (full body mode)" if is_full_body else sel["label"]
        st.markdown(f"**Query — {label_text}**")
        st.image(st.session_state.cropped_image, use_container_width=True)

    with col_info:
        chip_text = "Step 2 · Results" if is_full_body else f"Step 4 · Results — {sel['label']}"
        st.markdown(f'<div class="step-chip">{chip_text}</div>', unsafe_allow_html=True)
        el  = st.session_state.elapsed
        timing_keys = (
            [("CLIP","clip"), ("HNSW","hnsw"), ("BLIP ITM","blip")]
            if is_full_body
            else [("YOLOS","yolo"), ("CLIP","clip"), ("HNSW","hnsw"), ("BLIP ITM","blip")]
        )
        mc = st.columns(len(timing_keys))
        for col, (lbl, key) in zip(mc, timing_keys):
            with col:
                st.markdown(
                    f"<div class='metric-box'>"
                    f"<div class='metric-label'>{lbl}</div>"
                    f"<div class='metric-value'>{el.get(key,'–')} s</div>"
                    f"</div>", unsafe_allow_html=True)
        total = sum(v for v in el.values() if isinstance(v, float))
        st.markdown(
            f"<p style='color:#888;font-size:0.82rem'>"
            f"Total: <code>{total:.3f} s</code> &nbsp;|&nbsp; "
            f"Ablation: <code>{ablation.split('—')[0].strip()}</code> &nbsp;|&nbsp; "
            f"Re-rank: <code>{'on' if use_rerank else 'off'}</code> &nbsp;|&nbsp; "
            f"α = <code>{alpha}</code> &nbsp;|&nbsp; "
            f"K = <code>{top_k}</code> &nbsp;|&nbsp; "
            f"Mode: <code>{'full body' if is_full_body else 'crop'}</code></p>",
            unsafe_allow_html=True)

    st.markdown("---")

    # ── result display helper ──────────────────────────────────────────────────
    def render_result_grid(results: list[dict], show_itm: bool, title_html: str):
        st.markdown(title_html, unsafe_allow_html=True)
        n_cols = 5
        for row_start in range(0, len(results), n_cols):
            row  = results[row_start: row_start + n_cols]
            cols = st.columns(n_cols, gap="small")
            for col, item in zip(cols, row):
                with col:
                    gimg = load_gallery_image(item["image_name"])
                    if gimg:
                        st.image(gimg, use_container_width=True)
                    else:
                        st.markdown("*(image not found)*")

                    cap = get_caption(item["image_name"])
                    rank_num = item.get("post_rank") if show_itm else item.get("pre_rank")
                    score_line = (
                        f"<div style='font-size:10px;color:#aaa;font-family:Space Mono,monospace'>"
                        f"#{rank_num}</div>"
                        if rank_num else ""
                    )
                    score_line += f"<div class='sim-badge'>cos {item['cosine_score']:.3f}</div>"
                    if show_itm and item.get("itm_score") is not None:
                        score_line += f"<br><div class='rank-badge'>itm {item['itm_score']:.3f}</div>"
                        if item.get("blended_score") is not None:
                            score_line += f"<br><div class='rank-badge'>blend {item['blended_score']:.3f} (α={alpha})</div>"

                    st.markdown(
                        f"<div style='text-align:center'>"
                        f"<div style='font-size:10px;color:#777;font-family:Space Mono,monospace'>"
                        f"{item.get('item_id','')}</div>"
                        f"{score_line}"
                        f"<div style='font-size:10px;color:#666;margin-top:5px'>"
                        f"{cap[:45]}{'…' if len(cap)>45 else ''}</div>"
                        f"</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    # ── pre-reranking grid ─────────────────────────────────────────────────────
    render_result_grid(
        pre_results,
        show_itm=False,
        title_html=(
            "<div class='pre-rank-label'>📊 Pre-reranking — CLIP cosine similarity order</div>"
        ),
    )

    # ── post-reranking grid (only if BLIP was run) ─────────────────────────────
    if post_results:
        st.markdown(
            "<hr style='border-color:#333;margin:0.5rem 0'>",
            unsafe_allow_html=True,
        )
        render_result_grid(
            post_results,
            show_itm=True,
            title_html=(
                "<div class='post-rank-label'>🔁 Post-reranking — BLIP ITM score order</div>"
            ),
        )
    else:
        st.info("ℹ️ BLIP ITM re-ranking is disabled. Enable it in the sidebar to see post-reranking results.")

    # ── rank change summary ────────────────────────────────────────────────────
    if post_results:
        with st.expander("📈 Rank change summary (pre vs post)"):
            pre_map  = {r["image_name"]: r["pre_rank"]  for r in pre_results}
            post_map = {r["image_name"]: r.get("post_rank", i+1) for i, r in enumerate(post_results)}
            rows = []
            for r in post_results:
                name   = r["image_name"]
                pre_r  = pre_map.get(name, "–")
                post_r = post_map.get(name, "–")
                delta  = (pre_r - post_r) if isinstance(pre_r, int) and isinstance(post_r, int) else 0
                arrow  = "⬆️" if delta > 0 else ("⬇️" if delta < 0 else "➡️")
                rows.append({
                    "Image": name.split("/")[-1],
                    "CLIP rank": pre_r,
                    "BLIP rank": post_r,
                    "Δ": f"{arrow} {abs(delta)}" if delta != 0 else "—",
                    "cos": r["cosine_score"],
                    "itm": r.get("itm_score","–"),
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── raw metadata ───────────────────────────────────────────────────────────
    with st.expander("🔎 Raw result metadata"):
        st.json([{k:v for k,v in r.items() if k != "crop"} for r in pre_results])

    st.markdown("---")
    col_ns, col_cr, _ = st.columns([1,1,5])
    with col_ns:
        if st.button("⬅️ New search", use_container_width=True):
            reset(); st.rerun()
    with col_cr:
        if not is_full_body:
            if st.button("🔎 Change selection", use_container_width=True):
                st.session_state.stage   = "detect"
                st.session_state.pre_rerank  = None
                st.session_state.post_rerank = None
                st.rerun()
