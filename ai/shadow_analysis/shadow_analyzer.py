"""
Acoustic shadow analysis for side-scan sonar imagery.
Detects acoustic dropouts/shadows adjacent to bright sonar reflections to validate physical object presence.
"""

import time
from typing import Optional, Dict, Any, List
import cv2
import numpy as np


class ShadowAnalysisResult:
    """Holds quantitative metrics for acoustic shadow regions."""

    def __init__(self):
        self.shadow_score = 0.0
        self.object_intensity = 0.0
        self.shadow_intensity = 0.0
        self.background_intensity = 0.0
        self.object_area_px = 0
        self.shadow_area_px = 0
        self.shadow_to_object_ratio = 0.0
        self.shadow_length_px = 0
        self.object_width_px = 0
        self.orientation_deg = 0.0
        self.contrast_ratio = 0.0
        self.height_estimate = "Estimated height unavailable — requires sonar altitude/range metadata"
        self.shadow_bbox = None
        self.shadow_mask = None
        self.analysis_mode = "STANDARD"
        self.elapsed_ms = 0.0

    def to_dict(self) -> dict:
        return {
            "shadow_score": round(self.shadow_score, 3),
            "object_intensity": round(self.object_intensity, 1),
            "shadow_intensity": round(self.shadow_intensity, 1),
            "background_intensity": round(self.background_intensity, 1),
            "object_area_px": self.object_area_px,
            "shadow_area_px": self.shadow_area_px,
            "shadow_to_object_ratio": round(self.shadow_to_object_ratio, 3),
            "shadow_length_px": self.shadow_length_px,
            "object_width_px": self.object_width_px,
            "orientation_deg": round(self.orientation_deg, 1),
            "contrast_ratio": round(self.contrast_ratio, 3),
            "height_estimate": self.height_estimate,
            "shadow_bbox": self.shadow_bbox,
            "analysis_mode": self.analysis_mode,
            "elapsed_ms": self.elapsed_ms,
        }


class AcousticShadowAnalyzer:
    """Evaluates acoustic shadow consistency based on sensor geometry and reflection contrast."""

    SHADOW_DARK_THRESHOLD = 60
    OBJECT_BRIGHT_THRESHOLD = 150
    SEARCH_RADIUS_FACTOR = 3.0

    def analyze(self, img: np.ndarray, bbox: list,
                sonar_metadata: Optional[dict] = None) -> ShadowAnalysisResult:
        t0 = time.time()
        result = ShadowAnalysisResult()

        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = img.astype(np.float32)

        h, w = gray.shape
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            result.analysis_mode = "INVALID_BBOX"
            return result

        obj_region = gray[y1:y2, x1:x2]
        result.object_intensity = float(obj_region.mean())
        result.object_area_px = (x2 - x1) * (y2 - y1)
        result.object_width_px = x2 - x1

        bg_samples = []
        if y1 > 20 and x1 > 20:
            bg_samples.append(gray[:y1, :x1])
        if y2 < h - 20 and x1 > 20:
            bg_samples.append(gray[y2:min(h, y2 + 50), :x1])

        if bg_samples:
            bg_all = np.concatenate([b.ravel() for b in bg_samples])
            result.background_intensity = float(bg_all.mean()) if len(bg_all) > 0 else 80.0
        else:
            result.background_intensity = 80.0

        nadir_x = w // 2
        obj_cx = (x1 + x2) // 2
        shadow_dir = 1 if obj_cx > nadir_x else -1

        obj_w = x2 - x1
        search_w = int(obj_w * self.SEARCH_RADIUS_FACTOR)

        if shadow_dir > 0:
            sx1 = x2
            sx2 = min(w - 1, x2 + search_w)
        else:
            sx1 = max(0, x1 - search_w)
            sx2 = x1

        sy1 = max(0, y1 - obj_w // 4)
        sy2 = min(h - 1, y2 + obj_w // 4)

        if sx2 > sx1 and sy2 > sy1:
            shadow_search_region = gray[sy1:sy2, sx1:sx2]
            dark_mask = shadow_search_region < self.SHADOW_DARK_THRESHOLD
            result.shadow_area_px = int(dark_mask.sum())

            if result.shadow_area_px > 50:
                shadow_intensities = shadow_search_region[dark_mask]
                result.shadow_intensity = float(shadow_intensities.mean())

                dark_cols = np.where(dark_mask.any(axis=0))[0]
                if len(dark_cols) > 0:
                    result.shadow_length_px = int(dark_cols.max() - dark_cols.min() + 1)
                    dark_rows = np.where(dark_mask.any(axis=1))[0]
                    if len(dark_rows) > 0:
                        result.shadow_bbox = [
                            sx1 + int(dark_cols.min()),
                            sy1 + int(dark_rows.min()),
                            sx1 + int(dark_cols.max()),
                            sy1 + int(dark_rows.max()),
                        ]

                full_shadow_mask = np.zeros_like(gray, dtype=np.uint8)
                shadow_region_mask = (gray[sy1:sy2, sx1:sx2] < self.SHADOW_DARK_THRESHOLD).astype(np.uint8) * 255
                full_shadow_mask[sy1:sy2, sx1:sx2] = shadow_region_mask
                result.shadow_mask = full_shadow_mask
        else:
            result.shadow_intensity = result.background_intensity

        if result.object_area_px > 0:
            result.shadow_to_object_ratio = result.shadow_area_px / result.object_area_px

        if result.background_intensity > 0:
            obj_contrast = (result.object_intensity - result.background_intensity) / result.background_intensity
            shadow_contrast = (result.background_intensity - result.shadow_intensity) / result.background_intensity
            result.contrast_ratio = float(np.clip((obj_contrast + shadow_contrast) / 2, 0, 2))

        if result.object_area_px > 200:
            obj_mask = (gray[y1:y2, x1:x2] > self.OBJECT_BRIGHT_THRESHOLD).astype(np.uint8)
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if len(c) >= 5:
                    try:
                        (cx, cy), (ma, mi), angle = cv2.fitEllipse(c)
                        result.orientation_deg = float(angle)
                    except Exception:
                        pass

        score = 0.0
        if result.object_intensity > self.OBJECT_BRIGHT_THRESHOLD:
            obj_brightness_score = min(1.0, (result.object_intensity - self.OBJECT_BRIGHT_THRESHOLD) / 50.0)
        else:
            obj_brightness_score = 0.0
        score += 0.30 * obj_brightness_score

        if result.shadow_intensity < self.SHADOW_DARK_THRESHOLD and result.shadow_area_px > 50:
            shadow_dark_score = min(1.0, (self.SHADOW_DARK_THRESHOLD - result.shadow_intensity) / 30.0)
        else:
            shadow_dark_score = 0.0
        score += 0.30 * shadow_dark_score

        contrast_score = min(1.0, result.contrast_ratio)
        score += 0.20 * contrast_score

        if result.shadow_area_px > 100:
            score += 0.10

        if 0.3 < result.shadow_to_object_ratio < 5.0:
            score += 0.10

        result.shadow_score = round(float(np.clip(score, 0.0, 1.0)), 3)

        if sonar_metadata and sonar_metadata.get("altitude_m") and sonar_metadata.get("pixel_size_m"):
            altitude = sonar_metadata["altitude_m"]
            pixel_size = sonar_metadata["pixel_size_m"]
            shadow_m = result.shadow_length_px * pixel_size
            if shadow_m > 0 and altitude > 0:
                h_est = altitude * result.shadow_length_px / max(result.shadow_area_px, 1)
                result.height_estimate = f"~{h_est:.1f} m (geometry estimate)"
        else:
            result.height_estimate = (
                f"Relative estimate: {result.shadow_length_px} px length, "
                f"ratio {result.shadow_to_object_ratio:.2f}"
            )

        result.analysis_mode = "STANDARD"
        result.elapsed_ms = round((time.time() - t0) * 1000, 1)
        return result


def visualize_shadow(img: np.ndarray, bbox: list,
                     shadow_result: ShadowAnalysisResult) -> np.ndarray:
    """Overlays detection bounding box and highlighted acoustic shadow region."""
    if img.ndim == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()

    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)

    if shadow_result.shadow_bbox is not None:
        sx1, sy1_, sx2, sy2_ = [int(v) for v in shadow_result.shadow_bbox]
        overlay = vis.copy()
        cv2.rectangle(overlay, (sx1, sy1_), (sx2, sy2_), (0, 100, 255), -1)
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        cv2.rectangle(vis, (sx1, sy1_), (sx2, sy2_), (0, 80, 200), 2)
        cv2.putText(vis, "SHADOW", (sx1, max(sy1_ - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 200), 1)

    score_text = f"Shadow: {shadow_result.shadow_score:.0%}"
    cv2.putText(vis, score_text, (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    return vis
