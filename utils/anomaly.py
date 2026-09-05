"""
Statistical anomaly detection utilities for sonar imagery.
Evaluates local intensity variance, texture entropy, and edge density against background baselines.
"""

import time
from typing import Optional, Dict, Any, List
import cv2
import numpy as np

LOW_THRESH = 0.35
HIGH_THRESH = 0.60

METHOD_LABEL = "Statistical intensity & texture baseline"
METHOD_NOTE = (
    "Anomaly score computed by measuring local intensity contrast, "
    "Shannon entropy, and edge density against the background baseline."
)


def _entropy(region: np.ndarray) -> float:
    """Calculate Shannon entropy of pixel intensity distribution."""
    hist = cv2.calcHist([region], [0], None, [32], [0, 256]).flatten()
    hist = hist[hist > 0]
    p = hist / hist.sum()
    return float(-np.sum(p * np.log2(p + 1e-9)))


def _edge_density(region: np.ndarray) -> float:
    """Calculate edge pixel ratio using Canny detection."""
    edges = cv2.Canny(region, 30, 100)
    return float(edges.mean() / 255.0)


def score_region(gray: np.ndarray, bbox: list, background_intensity: float) -> dict:
    """Scores a region of interest based on anomaly characteristics."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = gray.shape
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    region = gray[y1:y2, x1:x2]
    if region.size < 16:
        return {"anomaly_score": 0.0, "status": "NORMAL", "components": {}}

    region_mean = float(region.mean())
    region_std = float(region.std())

    # Deviation metrics
    intensity_dev = min(1.0, abs(region_mean - background_intensity) / max(background_intensity, 1.0))
    entropy_score = min(1.0, _entropy(region) / 5.0)
    edge_score = min(1.0, _edge_density(region) * 5.0)
    contrast_score = min(1.0, region_std / 60.0)

    score = (
        intensity_dev * 0.35 +
        entropy_score * 0.30 +
        edge_score * 0.20 +
        contrast_score * 0.15
    )
    score = round(min(1.0, max(0.0, float(score))), 3)

    if score >= HIGH_THRESH:
        status = "ANOMALOUS"
    elif score >= LOW_THRESH:
        status = "UNCERTAIN"
    else:
        status = "NORMAL"

    return {
        "anomaly_score": score,
        "status": status,
        "components": {
            "intensity_deviation": round(intensity_dev, 3),
            "texture_entropy": round(entropy_score, 3),
            "edge_density": round(edge_score, 3),
            "local_contrast": round(contrast_score, 3),
        },
    }


def analyze_anomaly(img: np.ndarray,
                    bbox: Optional[list] = None,
                    background_sample: Optional[np.ndarray] = None) -> dict:
    """Analyze whole image or bounded region for unexpected acoustic features."""
    t0 = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    h, w = gray.shape

    border_px = min(30, h // 8, w // 8)
    border_region = np.concatenate([
        gray[:border_px, :].flatten(),
        gray[-border_px:, :].flatten(),
        gray[:, :border_px].flatten(),
        gray[:, -border_px:].flatten(),
    ])
    background_intensity = float(border_region.mean()) if border_region.size else 128.0

    target_bbox = bbox if bbox else [0, 0, w, h]
    result = score_region(gray, target_bbox, background_intensity)
    elapsed_ms = round((time.time() - t0) * 1000, 1)

    return {
        "anomaly_score": result["anomaly_score"],
        "anomaly_score_pct": round(result["anomaly_score"] * 100, 1),
        "status": result["status"],
        "is_anomalous": result["status"] == "ANOMALOUS",
        "background_intensity": round(background_intensity, 1),
        "components": result["components"],
        "method": METHOD_LABEL,
        "method_note": METHOD_NOTE,
        "mode": "DEMO_STATISTICAL",
        "elapsed_ms": elapsed_ms,
        "reasons": _reasons(result),
    }


def _reasons(result: dict) -> list:
    reasons = []
    c = result.get("components", {})
    if c.get("intensity_deviation", 0) > 0.5:
        reasons.append("Elevated intensity divergence from seabed background")
    if c.get("texture_entropy", 0) > 0.6:
        reasons.append("Unusual texture entropy")
    if c.get("edge_density", 0) > 0.4:
        reasons.append("Dense structural edge patterns")
    if c.get("local_contrast", 0) > 0.6:
        reasons.append("High local contrast variation")
    if not reasons:
        reasons.append("Nominal acoustic reflection")
    return reasons
