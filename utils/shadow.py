"""
Acoustic shadow extraction and physical dimension estimation.
Evaluates the acoustic shadow cast beyond targets relative to the central sonar nadir track.
"""

import time
from typing import Optional, Dict, Any, List
import cv2
import numpy as np


def analyze_shadow(
    img: np.ndarray,
    bbox: list,
    sonar_metadata: Optional[dict] = None,
) -> dict:
    """Analyze acoustic shadow geometry and contrast for candidate target."""
    t0 = time.time()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    h, w = gray.shape
    x1, y1, x2, y2 = [int(v) for v in bbox]

    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    obj_w = x2 - x1
    obj_h = y2 - y1

    if obj_w < 5 or obj_h < 5:
        return _no_shadow("Object bounding box too compact for shadow resolution", time.time() - t0)

    # Object region intensity
    obj_region = gray[y1:y2, x1:x2]
    object_intensity = float(obj_region.mean())

    # Surrounding background intensity
    pad = max(5, obj_h // 2)
    bg_y1 = max(0, y1 - pad)
    bg_y2 = min(h, y2 + pad)
    bg_x1 = max(0, x1 - pad)
    bg_x2 = min(w, x2 + pad)

    bg_mask = np.ones(gray.shape, dtype=bool)
    bg_mask[y1:y2, x1:x2] = False
    bg_mask[:bg_y1, :] = False
    bg_mask[bg_y2:, :] = False
    bg_mask[:, :bg_x1] = False
    bg_mask[:, bg_x2:] = False

    background_vals = gray[bg_mask]
    background_intensity = float(background_vals.mean()) if background_vals.size > 0 else 128.0

    # Nadir geometry determines shadow orientation
    img_centre_x = w // 2
    obj_centre_x = (x1 + x2) // 2

    if obj_centre_x < img_centre_x:
        shadow_x1 = max(0, x1 - obj_w * 2)
        shadow_x2 = x1
    else:
        shadow_x1 = x2
        shadow_x2 = min(w, x2 + obj_w * 2)

    shadow_y1 = y1
    shadow_y2 = y2

    if shadow_x2 <= shadow_x1 or shadow_y2 <= shadow_y1:
        return _no_shadow("Shadow region extends beyond frame", time.time() - t0)

    shadow_region = gray[shadow_y1:shadow_y2, shadow_x1:shadow_x2]
    shadow_intensity = float(shadow_region.mean())

    obj_is_bright = object_intensity > background_intensity * 1.15
    shadow_is_dark = shadow_intensity < background_intensity * 0.85
    contrast_ratio = (object_intensity - shadow_intensity) / max(object_intensity, 1.0)

    if not obj_is_bright:
        return _no_shadow(
            f"Insufficient object contrast (obj={object_intensity:.1f}, bg={background_intensity:.1f})",
            time.time() - t0,
            object_intensity=object_intensity,
            shadow_intensity=shadow_intensity,
            background_intensity=background_intensity,
        )

    dark_score = max(0.0, min(1.0, (background_intensity - shadow_intensity) / max(background_intensity, 1) * 3))
    bright_score = max(0.0, min(1.0, (object_intensity - background_intensity) / max(background_intensity, 1) * 3))
    contrast_score = max(0.0, min(1.0, contrast_ratio * 2))

    shadow_area_px = shadow_region.size
    object_area_px = obj_region.size
    area_ratio = min(1.0, shadow_area_px / max(object_area_px, 1))

    shadow_score = (
        dark_score * 0.30 +
        bright_score * 0.25 +
        contrast_score * 0.25 +
        area_ratio * 0.10 +
        0.10
    )
    shadow_score = round(min(1.0, max(0.0, shadow_score)), 3)
    shadow_detected = shadow_is_dark and shadow_score > 0.25
    shadow_length_px = shadow_x2 - shadow_x1

    height_estimate = _estimate_height(shadow_length_px, obj_h, sonar_metadata)
    elapsed_ms = round((time.time() - t0) * 1000, 1)

    return {
        "shadow_detected": shadow_detected,
        "shadow_score": shadow_score if shadow_detected else 0.0,
        "shadow_area_px": shadow_area_px,
        "shadow_length_px": shadow_length_px,
        "object_area_px": object_area_px,
        "object_intensity": round(object_intensity, 1),
        "shadow_intensity": round(shadow_intensity, 1),
        "background_intensity": round(background_intensity, 1),
        "shadow_to_object_ratio": round(shadow_area_px / max(object_area_px, 1), 3),
        "height_estimate": height_estimate,
        "shadow_bbox": [shadow_x1, shadow_y1, shadow_x2, shadow_y2] if shadow_detected else None,
        "shadow_side": "left" if obj_centre_x >= img_centre_x else "right",
        "status": "detected" if shadow_detected else "not_reliable",
        "note": "" if shadow_detected else "Low confidence shadow signature",
        "elapsed_ms": elapsed_ms,
    }


def _no_shadow(reason: str, elapsed: float, **kwargs) -> dict:
    return {
        "shadow_detected": False,
        "shadow_score": None,
        "shadow_area_px": 0,
        "shadow_length_px": 0,
        "object_area_px": 0,
        "object_intensity": kwargs.get("object_intensity", 0.0),
        "shadow_intensity": kwargs.get("shadow_intensity", 0.0),
        "background_intensity": kwargs.get("background_intensity", 0.0),
        "shadow_to_object_ratio": 0.0,
        "height_estimate": "Unavailable — no distinct acoustic shadow",
        "shadow_bbox": None,
        "shadow_side": None,
        "status": "not_reliable",
        "note": reason,
        "elapsed_ms": round(elapsed * 1000, 1),
    }


def _estimate_height(shadow_len_px: int, obj_height_px: int,
                     metadata: Optional[dict]) -> str:
    """Calculates physical relief estimate from acoustic shadow extension."""
    if metadata is None:
        return "Unavailable — requires altitude and slant range telemetry"

    altitude = metadata.get("altitude_m")
    slant_range_m = metadata.get("slant_range_m")
    pixel_size_m = metadata.get("pixel_size_m")

    if not all([altitude, slant_range_m, pixel_size_m]):
        return "Unavailable — incomplete telemetry metadata"

    try:
        shadow_len_m = shadow_len_px * pixel_size_m
        h = altitude * shadow_len_m / max(slant_range_m, 0.01)
        return f"~{h:.2f} m (geometric estimate)"
    except Exception:
        return "Unavailable — geometric estimation error"
