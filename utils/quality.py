"""
Image quality evaluation metrics for acoustic sensor data.
Measures focus/sharpness, contrast standard deviation, noise variance, and signal dropout.
"""

from typing import Dict, Any, List
import cv2
import numpy as np


def assess_quality(img: np.ndarray) -> dict:
    """Compute quantitative quality indicators for acoustic imagery."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray_f = gray.astype(np.float32)

    # Sharpness via Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    # Contrast via standard deviation
    contrast = float(gray_f.std())
    brightness = float(gray_f.mean())

    # High-frequency residual as noise approximation
    smoothed = cv2.GaussianBlur(gray_f, (5, 5), 0)
    noise_residual = np.abs(gray_f - smoothed)
    noise_estimate = float(noise_residual.mean())

    # Column dropout detection (transducer blanking or sync loss)
    col_means = gray_f.mean(axis=0)
    dropout_cols = int(np.sum(col_means < 5))
    missing_pct = round(dropout_cols / gray.shape[1] * 100, 2)

    reasons = []
    score_components = []

    sharpness_score = min(1.0, sharpness / 100.0)
    score_components.append(sharpness_score * 0.30)
    if sharpness < 10:
        reasons.append(f"Subdued sharpness ({sharpness:.1f})")
    elif sharpness < 30:
        reasons.append(f"Moderate sharpness ({sharpness:.1f})")

    contrast_score = min(1.0, contrast / 60.0)
    score_components.append(contrast_score * 0.25)
    if contrast < 10:
        reasons.append(f"Low contrast range ({contrast:.1f})")
    elif contrast < 20:
        reasons.append(f"Subdued contrast ({contrast:.1f})")

    brightness_score = max(0.0, 1.0 - abs(brightness - 128) / 128.0)
    score_components.append(brightness_score * 0.20)
    if brightness < 20:
        reasons.append(f"Low mean intensity ({brightness:.1f})")
    elif brightness > 235:
        reasons.append(f"High saturation ({brightness:.1f})")

    noise_score = max(0.0, 1.0 - noise_estimate / 30.0)
    score_components.append(noise_score * 0.15)
    if noise_estimate > 20:
        reasons.append(f"Elevated acoustic noise ({noise_estimate:.1f})")

    missing_score = max(0.0, 1.0 - missing_pct / 20.0)
    score_components.append(missing_score * 0.10)
    if missing_pct > 10:
        reasons.append(f"Column dropout ({missing_pct:.1f}%)")

    total_score = float(sum(score_components))

    if total_score >= 0.72:
        grade = "GOOD"
    elif total_score >= 0.45:
        grade = "ACCEPTABLE"
    else:
        grade = "POOR"

    return {
        "quality": grade,
        "score": round(total_score, 3),
        "reasons": reasons,
        "metrics": {
            "sharpness": round(sharpness, 2),
            "contrast": round(contrast, 2),
            "mean_brightness": round(brightness, 2),
            "noise_estimate": round(noise_estimate, 2),
            "missing_data_pct": round(missing_pct, 2),
        },
    }
