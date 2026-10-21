"""
Defective Product Detection System - Backend
============================================
Python implementation of the 8-stage pipeline used in the web prototype:

  1 Acquisition -> 2 Enhancement -> 3 Restoration -> 4 Preprocessing
  -> 5 Segmentation -> 6 Feature extraction -> 7 Classification -> 8 Output

Digital Image Processing (DIP): stages 1-6   (OpenCV / NumPy)
Machine Learning (ML):          stages 7-8   (rule-based scorer, or a trained
                                              Random Forest from scikit-learn)

Install:
    pip install opencv-python-headless numpy flask scikit-learn joblib

Usage:
    # 1) Inspect a single image (rule-based, compared against a normal reference)
    python inspection_backend.py inspect part.png --reference normal.png

    # 2) Train a Random Forest on a labelled dataset
    #    dataset/non_defective/*.png   dataset/defective/*.png
    python inspection_backend.py train dataset/ --reference normal.png

    # 3) Run the REST API  (POST an image to /api/inspect)
    python inspection_backend.py serve --reference normal.png --port 5000
"""

import argparse
import base64
import json
import os
import sys

import cv2
import numpy as np

IMG_SIZE = 200                       # fixed working size (matches the web prototype)
MODEL_PATH = "defect_model.joblib"   # created by the `train` command
REF_PATH = "reference_profile.json"  # created from the --reference image
DEFECT_THRESHOLD = 28.0              # rule-based accept/reject threshold (0-100)
FEATURE_NAMES = ["edge_density", "intensity_std", "fill_ratio", "circularity"]


# --------------------------------------------------------------------------- #
# Stages 1-6: Digital Image Processing
# --------------------------------------------------------------------------- #
def acquire(source):
    """Stage 1 - Acquisition. Accepts a file path or raw image bytes."""
    if isinstance(source, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(source, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read image")
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def stretch(g):
    """Min-max normalisation to 0-255."""
    lo, hi = float(g.min()), float(g.max())
    return (g - lo) / max(1.0, hi - lo) * 255.0


def enhance(gray):
    """Stage 2 - Enhancement: noise reduction + contrast stretch."""
    blurred = cv2.blur(gray, (3, 3))
    return stretch(blurred), blurred


def restore(enhanced, blurred, amount=1.15):
    """Stage 3 - Restoration: unsharp mask to recover edge clarity."""
    return np.clip(enhanced + amount * (enhanced - blurred), 0, 255)


def preprocess(restored):
    """Stage 4 - Preprocessing: grayscale normalisation (0-255)."""
    return stretch(restored)


def segment(pre):
    """Stage 5 - Segmentation: Otsu threshold, minority class = product."""
    u8 = pre.astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    below = int((u8 < thr).sum())
    object_darker = below <= u8.size / 2
    mask = (u8 < thr) if object_darker else (u8 >= thr)
    mask = mask.astype(np.uint8)
    # Clean-up: remove speckle/thin borders, keep only the largest blob (the
    # product) and fill its interior so the hole/thread does not break the mask.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [biggest], -1, 1, thickness=cv2.FILLED)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        bbox = (0, 0, IMG_SIZE, IMG_SIZE)
    else:
        bbox = (int(xs.min()), int(ys.min()),
                int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    return mask, bbox, float(thr)


def sobel_magnitude(pre):
    gx = cv2.Sobel(pre, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(pre, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def extract_features(pre, mask, bbox):
    """Stage 6 - Feature extraction: edge, texture, and shape descriptors."""
    mag = sobel_magnitude(pre)
    inside = mask.astype(bool)
    area = int(inside.sum())
    if area == 0:
        return dict.fromkeys(FEATURE_NAMES, 0.0), mag

    edge_density = float(mag[inside].mean())
    intensity_std = float(pre[inside].std())
    _, _, bw, bh = bbox
    fill_ratio = area / float(bw * bh)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours) or 1.0
    circularity = min(1.5, 4.0 * np.pi * area / (perimeter ** 2))

    return {
        "edge_density": edge_density,
        "intensity_std": intensity_std,
        "fill_ratio": float(fill_ratio),
        "circularity": float(circularity),
    }, mag


def run_dip_pipeline(source):
    """Run stages 1-6 and return every intermediate result."""
    img = acquire(source)
    gray = to_gray(img)
    enhanced, blurred = enhance(gray)
    restored = restore(enhanced, blurred)
    pre = preprocess(restored)
    mask, bbox, thr = segment(pre)
    feats, mag = extract_features(pre, mask, bbox)
    return {
        "image": img, "enhanced": enhanced, "restored": restored,
        "preprocessed": pre, "mask": mask, "bbox": bbox,
        "otsu_threshold": thr, "edges": mag, "features": feats,
    }


# --------------------------------------------------------------------------- #
# Stages 7-8: Classification and Output
# --------------------------------------------------------------------------- #
def build_reference(normal_image_path, save_to=REF_PATH):
    """Feature profile of a known-good product, used by the rule-based scorer."""
    feats = run_dip_pipeline(normal_image_path)["features"]
    with open(save_to, "w") as f:
        json.dump(feats, f, indent=2)
    return feats


def load_reference(path=REF_PATH):
    with open(path) as f:
        return json.load(f)


def rule_based_classify(feats, ref):
    """Transparent weighted deviation score vs. the normal reference profile."""
    d_edge = (feats["edge_density"] - ref["edge_density"]) / max(1.0, ref["edge_density"])
    d_var = (feats["intensity_std"] - ref["intensity_std"]) / max(1.0, ref["intensity_std"])
    d_fill = abs(feats["fill_ratio"] - ref["fill_ratio"]) / max(0.05, ref["fill_ratio"])
    d_circ = abs(feats["circularity"] - ref["circularity"]) / max(0.05, ref["circularity"])
    score = 100 * (0.40 * max(0, d_edge) + 0.20 * max(0, d_var)
                   + 0.22 * d_fill + 0.18 * d_circ)
    score = float(np.clip(score, 0, 100))
    defective = score >= DEFECT_THRESHOLD
    conf = (min(99, 55 + score * 0.5) if defective
            else min(99, 55 + (DEFECT_THRESHOLD - score) * 1.6))
    return {"method": "rule_based", "defect_score": round(score, 1),
            "verdict": "DEFECTIVE" if defective else "NON-DEFECTIVE",
            "confidence": round(conf, 1)}


def model_classify(feats, model):
    """Trained scikit-learn model (Random Forest) on the feature vector."""
    x = np.array([[feats[n] for n in FEATURE_NAMES]])
    p_def = float(model.predict_proba(x)[0][list(model.classes_).index(1)])
    defective = p_def >= 0.5
    return {"method": "random_forest", "defect_score": round(p_def * 100, 1),
            "verdict": "DEFECTIVE" if defective else "NON-DEFECTIVE",
            "confidence": round((p_def if defective else 1 - p_def) * 100, 1)}


def inspect(source, ref=None, model=None):
    """Full pipeline: stages 1-8. Returns a JSON-serialisable result."""
    dip = run_dip_pipeline(source)
    feats = dip["features"]
    if model is not None:
        result = model_classify(feats, model)
    elif ref is not None:
        result = rule_based_classify(feats, ref)
    else:
        raise ValueError("Provide a reference profile or a trained model")
    result["features"] = {k: round(v, 4) for k, v in feats.items()}
    result["otsu_threshold"] = round(dip["otsu_threshold"], 1)
    result["roi_bbox"] = dict(zip(("x", "y", "w", "h"), dip["bbox"]))
    return result, dip


# --------------------------------------------------------------------------- #
# Machine-learning training (optional, replaces the rule-based scorer)
# --------------------------------------------------------------------------- #
def train_model(dataset_dir, model_path=MODEL_PATH):
    """
    Expects:  dataset_dir/non_defective/*.(png|jpg)   -> label 0
              dataset_dir/defective/*.(png|jpg)       -> label 1
    Reports accuracy, precision, recall, F1 and a confusion matrix on a
    held-out test split, then saves the model.
    """
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    X, y = [], []
    for label, folder in ((0, "non_defective"), (1, "defective")):
        folder_path = os.path.join(dataset_dir, folder)
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                feats = run_dip_pipeline(os.path.join(folder_path, fname))["features"]
                X.append([feats[n] for n in FEATURE_NAMES])
                y.append(label)
    X, y = np.array(X), np.array(y)
    print(f"Loaded {len(y)} images ({int((y == 0).sum())} non-defective, "
          f"{int((y == 1).sum())} defective)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42,
                                   class_weight="balanced")
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(classification_report(y_te, pred,
                                target_names=["non_defective", "defective"]))
    print("Confusion matrix [rows=true, cols=predicted]:\n",
          confusion_matrix(y_te, pred))
    joblib.dump(model, model_path)
    print("Saved model ->", model_path)


def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        import joblib
        return joblib.load(path)
    return None


# --------------------------------------------------------------------------- #
# REST API
# --------------------------------------------------------------------------- #
def _b64_png(arr):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def create_app(ref, model):
    from flask import Flask, jsonify, request
    app = Flask(__name__)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok",
                       classifier="random_forest" if model else "rule_based")

    @app.post("/api/inspect")
    def api_inspect():
        if "image" not in request.files:
            return jsonify(error="Send an image as multipart field 'image'"), 400
        try:
            result, dip = inspect(request.files["image"].read(), ref, model)
        except ValueError as e:
            return jsonify(error=str(e)), 400
        if request.args.get("stages") == "1":   # optionally return stage images
            result["stages"] = {
                "enhanced": _b64_png(dip["enhanced"]),
                "restored": _b64_png(dip["restored"]),
                "preprocessed": _b64_png(dip["preprocessed"]),
                "segmentation_mask": _b64_png(dip["mask"] * 255),
                "edges": _b64_png(dip["edges"] / max(1.0, dip["edges"].max()) * 255),
            }
        return jsonify(result)

    return app


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Defective product detection backend")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="inspect one image")
    p.add_argument("image")
    p.add_argument("--reference", help="image of a known non-defective product")
    p.add_argument("--save-stages", help="folder to save intermediate stage images")

    p = sub.add_parser("train", help="train a Random Forest on a labelled folder")
    p.add_argument("dataset")
    p.add_argument("--reference")

    p = sub.add_parser("serve", help="start the REST API")
    p.add_argument("--reference")
    p.add_argument("--port", type=int, default=5000)

    if len(sys.argv) == 1:
        print("=" * 60)
        print("Defective Product Detection System - 8-Stage DIP Pipeline")
        print("=" * 60)
        sample_img = "WhatsApp Image 2026-09-14 at 8.25.23 AM.jpeg"
        if not os.path.exists(sample_img):
            for f in os.listdir("."):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    sample_img = f
                    break

        if os.path.exists(sample_img):
            print(f"[+] Running default inspection on: {sample_img}")
            ref = load_reference() if os.path.exists(REF_PATH) else build_reference(sample_img)
            model = load_model()
            result, dip = inspect(sample_img, ref, model)
            print("\n[INSPECTION RESULT]")
            print(json.dumps(result, indent=2))
            print(f"\nVerdict: {result.get('verdict')} (Score: {result.get('defect_score')}/100, Conf: {result.get('confidence')}%)")
            print("-" * 60)
            print("CLI options: 'inspect <image>', 'serve --port 5000', or 'train <dir>'")
            print("=" * 60)
            return

    args = ap.parse_args()

    ref = None
    if getattr(args, "reference", None):
        ref = build_reference(args.reference)
    elif os.path.exists(REF_PATH):
        ref = load_reference()

    if args.cmd == "train":
        train_model(args.dataset)
    elif args.cmd == "inspect":
        model = load_model()
        result, dip = inspect(args.image, ref, model)
        print(json.dumps(result, indent=2))
        if args.save_stages:
            os.makedirs(args.save_stages, exist_ok=True)
            for name in ("enhanced", "restored", "preprocessed"):
                cv2.imwrite(os.path.join(args.save_stages, f"{name}.png"),
                            np.clip(dip[name], 0, 255).astype(np.uint8))
            cv2.imwrite(os.path.join(args.save_stages, "mask.png"), dip["mask"] * 255)
            e = dip["edges"]
            cv2.imwrite(os.path.join(args.save_stages, "edges.png"),
                        (e / max(1.0, e.max()) * 255).astype(np.uint8))
    elif args.cmd == "serve":
        create_app(ref, load_model()).run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    sys.exit(main())