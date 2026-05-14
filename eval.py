"""
Batch Evaluation Script — Visual Product Search Engine
VR Course Final Project — Deliverable 3

Usage:
    python eval.py --top_k 10 --use_rerank --ablation C

What it does:
    Loops over all query images in the DeepFashion partition file,
    runs the full retrieval pipeline on each, and computes:
        Recall@K, NDCG@K, mAP@K  for K in {5, 10, 15}
    Reported as mean ± std over all queries.
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import hnswlib
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    BlipForImageTextRetrieval,
    BlipProcessor,
)

logging.getLogger("root").setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  ← set paths before running
# ══════════════════════════════════════════════════════════════════════════════

CHKPT_DIR      = Path("/kaggle/input/datasets/rrickyroger/checkpoints")
CAPTIONS_PATH  = Path("/kaggle/input/datasets/rrickyroger/captions/captions.json")
DATASET_ROOT   = Path("/kaggle/input/datasets/abhinavkishan123/deepfashion-inshop-dataset/Dataset")
PARTITION_FILE = DATASET_ROOT / "list_eval_partition.txt"
IMG_ROOT       = DATASET_ROOT

HNSW_INDEX    = CHKPT_DIR / "hnsw_index.bin"
METADATA_FILE = CHKPT_DIR / "index_metadata.pkl"
CLIP_CKPT     = CHKPT_DIR / "clip_finetuned.pt"

CONF_THRESHOLD = 0.25
MIN_BBOX_RATIO = 0.05
PADDING        = 0.08
FASHION_CATEGORIES = {
    'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan',
    'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit',
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def recall_at_k(retrieved_ids, true_id, k):
    return int(true_id in retrieved_ids[:k])

def ndcg_at_k(retrieved_ids, true_id, n_relevant, k):
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, rid in enumerate(retrieved_ids[:k])
        if rid == true_id
    )
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(n_relevant, k)))
    return dcg / ideal if ideal > 0 else 0.0

def ap_at_k(retrieved_ids, true_id, n_relevant, k):
    hits, sum_prec = 0, 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid == true_id:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / min(n_relevant, k) if n_relevant > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_models():
    print(f"Device: {DEVICE}")

    print("Loading CLIP...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained=None
    )
    clip_model.load_state_dict(
        torch.load(CLIP_CKPT, map_location=DEVICE, weights_only=False)
    )
    clip_model.eval().to(DEVICE)

    print("Loading YOLOS...")
    yolo_proc = AutoImageProcessor.from_pretrained("valentinafevu/yolos-fashionpedia")
    yolo_model = AutoModelForObjectDetection.from_pretrained(
        "valentinafevu/yolos-fashionpedia"
    ).to(DEVICE)
    yolo_model.eval()

    print("Loading BLIP ITM...")
    blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    blip_model = BlipForImageTextRetrieval.from_pretrained(
        "Salesforce/blip-itm-base-coco",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)
    blip_model.eval()

    print("Loading HNSW index...")
    with open(METADATA_FILE, "rb") as f:
        gallery_meta = pickle.load(f)
    index = hnswlib.Index(space="cosine", dim=512)
    index.load_index(str(HNSW_INDEX))
    index.set_ef(50)
    print(f"Index: {index.get_current_count()} items")

    print("Loading captions...")
    with open(CAPTIONS_PATH) as f:
        raw = json.load(f)
    captions = {}
    for k, v in raw.items():
        if "img/img/" in k:
            captions["img/img/" + k.split("img/img/")[-1]] = v

    return clip_model, clip_preprocess, yolo_proc, yolo_model, \
           blip_proc, blip_model, index, gallery_meta, captions


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE STEPS
# ══════════════════════════════════════════════════════════════════════════════

def yolo_auto_crop(image, yolo_proc, yolo_model):
    """Auto-selects highest-confidence fashion detection (no user input needed)."""
    img_w, img_h = image.size
    inputs = yolo_proc(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = yolo_model(**inputs)
    target_sizes = torch.tensor([[img_h, img_w]]).to(DEVICE)
    results = yolo_proc.post_process_object_detection(
        outputs, threshold=CONF_THRESHOLD, target_sizes=target_sizes
    )[0]

    best, best_score = None, -1
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        lname = yolo_model.config.id2label.get(label.item(), "")
        if lname not in FASHION_CATEGORIES:
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        if ((x2-x1)/img_w) < MIN_BBOX_RATIO or ((y2-y1)/img_h) < MIN_BBOX_RATIO:
            continue
        if score.item() > best_score:
            best_score = score.item()
            best = (x1, y1, x2, y2)

    if best is None:
        return image  # fallback to full image

    x1, y1, x2, y2 = best
    pw = int((x2-x1) * PADDING); ph = int((y2-y1) * PADDING)
    return image.crop((
        max(0, x1-pw), max(0, y1-ph),
        min(img_w, x2+pw), min(img_h, y2+ph)
    ))


def clip_embed(crop, clip_model, clip_preprocess):
    with torch.no_grad():
        t = clip_preprocess(crop).unsqueeze(0).to(DEVICE)
        emb = F.normalize(clip_model.encode_image(t), dim=-1)
    vec = emb.cpu().float().numpy()
    return vec / np.linalg.norm(vec)


def hnsw_retrieve(q_vec, index, gallery_meta, k):
    labels, dists = index.knn_query(q_vec, k=k)
    return [
        {**gallery_meta[i], "cosine_score": float(1 - d)}
        for i, d in zip(labels[0], dists[0])
    ]


def blip_rerank(crop, candidates, blip_proc, blip_model, captions):
    scored = []
    for c in candidates:
        img_name = c["image_name"]
        key = "img/img/" + img_name.split("img/img/")[-1] if "img/img/" in img_name else img_name
        caption = captions.get(key, "fashion clothing")
        inputs = blip_proc(
            images=crop, text=caption, return_tensors="pt", padding=True
        ).to(DEVICE)
        with torch.no_grad():
            score = torch.softmax(blip_model(**inputs).itm_score, dim=1)[0][1].item()
        scored.append({**c, "itm_score": score})
    return sorted(scored, key=lambda x: x["itm_score"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN EVAL LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k",      type=int,  default=15,
                        help="Max K to evaluate (evaluates at 5,10,15 up to this)")
    parser.add_argument("--use_rerank", action="store_true",
                        help="Enable BLIP ITM re-ranking")
    parser.add_argument("--ablation",   type=str,  default="C",
                        choices=["A","B","C"], help="Ablation config (A/B/C)")
    parser.add_argument("--limit",      type=int,  default=None,
                        help="Limit number of queries (for quick testing)")
    parser.add_argument("--output_csv", type=str,  default="eval_results.csv",
                        help="Path to save per-query results CSV")
    args = parser.parse_args()

    if args.ablation == "A":
        args.use_rerank = False
        print("Ablation A: vision-only CLIP, no re-ranking")
    elif args.ablation == "B":
        print("Ablation B: frozen CLIP + BLIP-2 re-ranking")
    else:
        print("Ablation C: fine-tuned CLIP + BLIP-2 re-ranking")

    K_VALUES = [k for k in [5, 10, 15] if k <= args.top_k]
    retrieve_k = max(K_VALUES)

    # Load everything
    (clip_model, clip_preprocess, yolo_proc, yolo_model,
     blip_proc, blip_model, index, gallery_meta, captions) = load_models()

    # Build item_id → count map (n_relevant per query)
    item_id_counts = {}
    for m in gallery_meta:
        item_id_counts[m["item_id"]] = item_id_counts.get(m["item_id"], 0) + 1

    # Load query partition
    query_df = pd.read_csv(
        PARTITION_FILE, skiprows=1, sep=r"\s+",
        names=["image_name", "item_id", "evaluation_status"]
    )
    query_df = query_df[query_df["evaluation_status"] == "query"].reset_index(drop=True)
    if args.limit:
        query_df = query_df.head(args.limit)
    print(f"\nEvaluating {len(query_df)} queries at K={K_VALUES}, re-rank={args.use_rerank}\n")

    # Per-query metric accumulators
    scores = {k: {"recall": [], "ndcg": [], "ap": []} for k in K_VALUES}
    rows   = []

    for _, row in tqdm(query_df.iterrows(), total=len(query_df), desc="Queries"):
        img_path = IMG_ROOT / row["image_name"]
        if not img_path.exists():
            continue
        true_id    = row["item_id"]
        n_relevant = item_id_counts.get(true_id, 1)

        try:
            image = Image.open(img_path).convert("RGB")
            crop  = yolo_auto_crop(image, yolo_proc, yolo_model)
            q_vec = clip_embed(crop, clip_model, clip_preprocess)
            cands = hnsw_retrieve(q_vec, index, gallery_meta, retrieve_k)

            if args.use_rerank:
                cands = blip_rerank(crop, cands, blip_proc, blip_model, captions)

            retrieved_ids = [c["item_id"] for c in cands]

            row_metrics = {"image_name": row["image_name"], "item_id": true_id}
            for k in K_VALUES:
                r  = recall_at_k(retrieved_ids, true_id, k)
                nd = ndcg_at_k(retrieved_ids, true_id, n_relevant, k)
                ap = ap_at_k(retrieved_ids, true_id, n_relevant, k)
                scores[k]["recall"].append(r)
                scores[k]["ndcg"].append(nd)
                scores[k]["ap"].append(ap)
                row_metrics[f"recall@{k}"]  = r
                row_metrics[f"ndcg@{k}"]    = nd
                row_metrics[f"map@{k}"]     = ap
            rows.append(row_metrics)

        except Exception as e:
            tqdm.write(f"  Skipped {row['image_name']}: {e}")
            continue

    # ── Print summary table ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Ablation {args.ablation}  |  Re-rank: {args.use_rerank}  |  N={len(rows)} queries")
    print(f"{'='*60}")
    print(f"{'Metric':<15} " + "  ".join(f"K={k:<5}" for k in K_VALUES))
    print("-"*60)
    for metric, key in [("Recall@K","recall"), ("NDCG@K","ndcg"), ("mAP@K","ap")]:
        vals = [scores[k][key] for k in K_VALUES]
        means = [np.mean(v) for v in vals]
        stds  = [np.std(v)  for v in vals]
        row_str = "  ".join(f"{m:.4f}±{s:.4f}" for m,s in zip(means, stds))
        print(f"{metric:<15} {row_str}")
    print(f"{'='*60}\n")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Per-query results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
