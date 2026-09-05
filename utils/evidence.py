"""
Evidence fusion module.
Combines detector confidence, shadow scores, regional anomaly scores, and image quality metrics into a composite score.
"""

from typing import Optional, Dict, Any

DEFAULT_WEIGHTS = {
    "detector": 0.45,
    "shadow": 0.20,
    "anomaly": 0.20,
    "image_quality": 0.15,
}

SEVERITY_HIGH = 0.70
SEVERITY_MEDIUM = 0.40

HIGH_RISK_CLASSES = {
    "fishing_net", "ghost_net", "metal_debris", "shipwreck",
    "container", "large_structure", "pipe_cable", "pipeline",
    "unknown_anomaly",
}


def compute_evidence(
    detector_confidence: float,
    shadow_score: Optional[float],
    anomaly_score: float,
    image_quality_score: float,
    weights: Optional[dict] = None,
) -> dict:
    """Calculates weighted evidence score and redistributes weights if shadow is absent."""
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)

    if shadow_score is None:
        extra = w.pop("shadow", 0.0)
        total_remaining = sum(w.values())
        if total_remaining > 0:
            for k in w:
                w[k] += extra * w[k] / total_remaining
        shadow_score_used = None
        shadow_contribution = None
    else:
        shadow_score_used = float(shadow_score)
        shadow_contribution = w.get("shadow", 0.20) * shadow_score_used

    score = (
        w.get("detector", 0.45) * float(detector_confidence) +
        w.get("anomaly", 0.20) * float(anomaly_score) +
        w.get("image_quality", 0.15) * float(image_quality_score)
    )
    if shadow_contribution is not None:
        score += shadow_contribution

    score = round(min(1.0, max(0.0, score)), 4)
    pct = round(score * 100, 1)

    breakdown = {
        "detector": {
            "score": round(detector_confidence, 3),
            "weight": w.get("detector", 0.45),
            "contribution": round(w.get("detector", 0.45) * detector_confidence, 4),
        },
        "shadow": {
            "score": shadow_score_used,
            "weight": w.get("shadow", 0.20) if shadow_score is not None else "redistributed",
            "contribution": round(shadow_contribution, 4) if shadow_contribution else "N/A",
        },
        "anomaly": {
            "score": round(anomaly_score, 3),
            "weight": w.get("anomaly", 0.20),
            "contribution": round(w.get("anomaly", 0.20) * anomaly_score, 4),
        },
        "image_quality": {
            "score": round(image_quality_score, 3),
            "weight": w.get("image_quality", 0.15),
            "contribution": round(w.get("image_quality", 0.15) * image_quality_score, 4),
        },
    }

    return {
        "evidence_score": score,
        "evidence_pct": pct,
        "breakdown": breakdown,
        "weights_used": w,
        "note": "Multi-signal weighted confidence aggregation.",
    }


def compute_severity(class_name: str,
                     evidence_score: float,
                     is_anomaly: bool = False,
                     object_area_px: int = 0) -> str:
    """Determines alert severity level from object class and composite evidence."""
    cn = (class_name or "").lower().replace(" ", "_")

    if is_anomaly:
        return "HIGH" if evidence_score >= SEVERITY_MEDIUM else "MEDIUM"

    if evidence_score >= SEVERITY_HIGH:
        sev = "HIGH"
    elif evidence_score >= SEVERITY_MEDIUM:
        sev = "MEDIUM"
    else:
        sev = "LOW"

    if cn in HIGH_RISK_CLASSES and sev == "LOW":
        sev = "MEDIUM"

    return sev


def get_severity_color(severity: str) -> str:
    """Color palette mapping for risk levels."""
    return {"HIGH": "#FF667A", "MEDIUM": "#F4C95D", "LOW": "#38D39F"}.get(severity, "#91B7C0")


def classify_manmade(class_name: str, evidence_score: float, anomaly_score: float) -> dict:
    """Heuristic categorization between anthropogenic debris and natural geomorphology."""
    natural = {
        "rock", "natural_formation", "sand_ripple", "seabed_feature",
        "bedform", "sediment", "coral"
    }
    artificial = {
        "fishing_net", "ghost_net", "plastic_debris", "metal_debris",
        "tire", "container", "bottle_object", "pipe_cable", "pipeline",
        "shipwreck", "other_debris", "unknown_anomaly"
    }
    cn = (class_name or "").lower().replace(" ", "_")

    if cn in natural:
        return {
            "type": "LIKELY_NATURAL",
            "is_manmade": False,
            "confidence": "class-based",
            "reason": f"Class '{class_name}' identified as natural seabed feature",
        }
    if cn in artificial:
        return {
            "type": "LIKELY_ARTIFICIAL",
            "is_manmade": True,
            "confidence": "class-based",
            "reason": f"Class '{class_name}' identified as anthropogenic debris",
        }
    if anomaly_score > 0.65:
        return {
            "type": "LIKELY_ARTIFICIAL",
            "is_manmade": True,
            "confidence": "anomaly-based",
            "reason": "Elevated anomaly score suggests non-natural origin",
        }
    return {
        "type": "UNCERTAIN",
        "is_manmade": None,
        "confidence": "insufficient",
        "reason": "Insufficient morphological evidence",
    }
