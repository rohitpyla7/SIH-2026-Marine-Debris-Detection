import os
import cv2
import yaml
import numpy as np
from pathlib import Path
from ai.shadow_analysis.shadow_analyzer import AcousticShadowAnalyzer

IMG_DIR = Path("./data/subpipe_hf_yolo/images/test")
LBL_DIR = Path("./data/subpipe_hf_yolo/labels/test")

analyzer = AcousticShadowAnalyzer()

positive_scores = []
negative_scores = []

for img_path in sorted(IMG_DIR.glob("*.png")):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    label_path = LBL_DIR / f"{img_path.stem}.txt"
    has_gt = label_path.exists() and label_path.read_text().strip() != ""

    try:
        result = analyzer.analyze(img)

        if isinstance(result, dict):
            score = result.get("shadow_score", result.get("score", 0.0))
        else:
            score = getattr(result, "shadow_score", getattr(result, "score", 0.0))

        score = float(score or 0.0)

        if has_gt:
            positive_scores.append(score)
        else:
            negative_scores.append(score)

    except Exception as e:
        print(f"ERROR {img_path.name}: {e}")

print()
print("===== SHADOW ANALYSIS TEST =====")
print(f"Positive images : {len(positive_scores)}")
print(f"Negative images : {len(negative_scores)}")

if positive_scores:
    print(f"Positive mean   : {np.mean(positive_scores):.4f}")
    print(f"Positive median : {np.median(positive_scores):.4f}")
    print(f"Positive min    : {np.min(positive_scores):.4f}")
    print(f"Positive max    : {np.max(positive_scores):.4f}")

if negative_scores:
    print(f"Negative mean   : {np.mean(negative_scores):.4f}")
    print(f"Negative median : {np.median(negative_scores):.4f}")
    print(f"Negative min    : {np.min(negative_scores):.4f}")
    print(f"Negative max    : {np.max(negative_scores):.4f}")

print("================================")
