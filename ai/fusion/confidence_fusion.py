"""
Confidence fusion and threat assessment logic.
Fuses detector confidence, acoustic shadow metrics, and regional anomaly signals into an aggregate score.
"""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any

DEFAULT_WEIGHTS = {
    "detector_weight": 0.45,
    "shadow_weight": 0.20,
    "texture_weight": 0.15,
    "shape_weight": 0.10,
    "anomaly_weight": 0.10,
}

HIGH_RISK_CLASSES = {"fishing_net", "shipwreck", "container", "pipe_cable", "pipeline"}
MEDIUM_RISK_CLASSES = {"metal_debris", "tire", "plastic_debris"}
LOW_RISK_CLASSES = {"bottle_object", "other_debris", "rock", "natural_formation"}


def load_fusion_config(config_path: str = "configs/confidence.yaml") -> dict:
    """Load fusion weights from configuration file or return defaults."""
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("fusion", DEFAULT_WEIGHTS)
    except Exception:
        return DEFAULT_WEIGHTS


def compute_evidence_score(
    detector_confidence: float,
    shadow_score: float = 0.0,
    anomaly_score: float = 0.0,
    image_quality_score: float = 1.0,
    texture_score: float = 0.5,
    shape_score: float = 0.5,
    weights: Optional[dict] = None,
) -> dict:
    """Calculate multi-modal weighted evidence score for detected candidates."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    w_det = weights.get("detector_weight", 0.45)
    w_shad = weights.get("shadow_weight", 0.20)
    w_tex = weights.get("texture_weight", 0.15)
    w_shape = weights.get("shape_weight", 0.10)
    w_anom = weights.get("anomaly_weight", 0.10)

    # In known detections, low anomaly indicates expected morphology
    anomaly_contrib = 1.0 - min(1.0, anomaly_score)
    quality_factor = max(0.5, image_quality_score)

    raw_score = (
        w_det * detector_confidence +
        w_shad * shadow_score +
        w_tex * texture_score +
        w_shape * shape_score +
        w_anom * anomaly_contrib
    )

    evidence_score = round(float(min(1.0, max(0.0, raw_score * quality_factor))), 3)

    return {
        "evidence_score": evidence_score,
        "evidence_pct": round(evidence_score * 100, 1),
        "breakdown": {
            "detector_contribution": round(w_det * detector_confidence, 3),
            "shadow_contribution": round(w_shad * shadow_score, 3),
            "texture_contribution": round(w_tex * texture_score, 3),
            "shape_contribution": round(w_shape * shape_score, 3),
            "anomaly_contribution": round(w_anom * anomaly_contrib, 3),
            "quality_factor": round(quality_factor, 3),
        },
        "label": "Fused Evidence Score",
        "weights_used": weights,
    }


def compute_anomaly_evidence_score(
    anomaly_score: float,
    shadow_score: float = 0.0,
    image_quality_score: float = 1.0,
) -> dict:
    """Calculate evidence score for unknown acoustic anomaly candidates."""
    quality_factor = max(0.5, image_quality_score)
    raw = 0.60 * anomaly_score + 0.40 * shadow_score
    evidence_score = round(float(min(1.0, raw * quality_factor)), 3)
    return {
        "evidence_score": evidence_score,
        "evidence_pct": round(evidence_score * 100, 1),
        "label": "Anomaly Evidence Score",
        "breakdown": {
            "anomaly_score_contribution": round(0.60 * anomaly_score, 3),
            "shadow_contribution": round(0.40 * shadow_score, 3),
            "quality_factor": round(quality_factor, 3),
        },
    }


def compute_severity(
    class_name: str,
    evidence_score: float,
    is_anomaly: bool = False,
    anomaly_score: float = 0.0,
    object_area_px: int = 0,
) -> str:
    """Assess severity level based on class risk profile and detection evidence."""
    cn = class_name.lower().replace(" ", "_")

    if is_anomaly or cn == "unknown_anomaly":
        if anomaly_score >= 0.80:
            return "HIGH"
        return "MEDIUM"

    if cn in HIGH_RISK_CLASSES:
        base = "HIGH"
    elif cn in MEDIUM_RISK_CLASSES:
        base = "MEDIUM"
    elif cn in LOW_RISK_CLASSES:
        base = "LOW"
    else:
        base = "MEDIUM"

    if evidence_score < 0.40:
        if base == "HIGH":
            base = "MEDIUM"
        elif base == "MEDIUM":
            base = "LOW"

    if object_area_px > 10000 and base == "MEDIUM":
        base = "HIGH"

    return base


def get_severity_color(severity: str) -> str:
    """Color codes for visualization badges."""
    colors = {
        "HIGH": "#FF4B4B",
        "MEDIUM": "#FFA500",
        "LOW": "#00CC66",
        "UNKNOWN": "#888888",
    }
    return colors.get(severity.upper(), "#888888")
