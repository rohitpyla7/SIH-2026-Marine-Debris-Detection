"""
Sonar imagery preprocessing pipeline and quality assessment.
Applies histogram equalization, denoising, and contrast enhancements to single/multi-channel acoustic images.
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np
import yaml


def load_config(config_path: str = "configs/preprocessing.yaml") -> dict:
    """Load preprocessing configuration parameters."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ImageQualityAssessor:
    """Computes basic quality metrics for incoming sonar frames."""

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            self.blur_thr = 100.0
            self.contrast_thr = 30.0
            self.noise_thr = 25.0
            self.min_dyn = 50
        else:
            qa = config.get("quality_assessment", {})
            self.blur_thr = qa.get("blur_threshold", 100.0)
            self.contrast_thr = qa.get("contrast_threshold", 30.0)
            self.noise_thr = qa.get("noise_threshold", 25.0)
            self.min_dyn = qa.get("min_dynamic_range", 50)

    def assess(self, img: np.ndarray) -> dict:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gray = gray.astype(np.float32)

        # Sharpness via Laplacian variance
        lap_var = float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())
        contrast = float(gray.std())
        dyn_range = int(gray.max() - gray.min())

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_est = float(np.abs(gray - blurred).mean())
        mean_brightness = float(gray.mean())

        reasons = []
        score = 1.0

        if lap_var < self.blur_thr:
            reasons.append(f"Low sharpness ({lap_var:.1f})")
            score -= 0.25

        if contrast < self.contrast_thr:
            reasons.append(f"Low contrast ({contrast:.1f})")
            score -= 0.25

        if noise_est > self.noise_thr:
            reasons.append(f"Elevated noise ({noise_est:.1f})")
            score -= 0.20

        if dyn_range < self.min_dyn:
            reasons.append(f"Restricted dynamic range ({dyn_range})")
            score -= 0.20

        score = max(0.0, min(1.0, score))

        if score >= 0.75:
            quality = "GOOD"
        elif score >= 0.50:
            quality = "ACCEPTABLE"
        else:
            quality = "POOR"

        return {
            "quality": quality,
            "score": round(score, 3),
            "reasons": reasons,
            "metrics": {
                "sharpness": round(lap_var, 2),
                "contrast": round(contrast, 2),
                "noise_estimate": round(noise_est, 2),
                "dynamic_range": dyn_range,
                "mean_brightness": round(mean_brightness, 2),
            },
            "warning": len(reasons) > 0,
        }


class SonarPreprocessor:
    """Configurable pipeline for acoustic image normalization and enhancement."""

    def __init__(self, config_path: str = "configs/preprocessing.yaml"):
        try:
            self.config = load_config(config_path)
        except Exception:
            self.config = {"pipeline": {"steps": []}}

        self.quality_assessor = ImageQualityAssessor(self.config)

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        mn, mx = img.min(), img.max()
        if mx - mn < 1:
            return img
        return ((img.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)

    def _clahe(self, img: np.ndarray, clip_limit: float = 2.0,
               tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        return clahe.apply(img)

    def _denoise(self, img: np.ndarray, method: str = "gaussian",
                 kernel: int = 3) -> np.ndarray:
        if method == "gaussian":
            return cv2.GaussianBlur(img, (kernel, kernel), 0)
        elif method == "median":
            return cv2.medianBlur(img, kernel)
        elif method == "bilateral":
            return cv2.bilateralFilter(img, kernel, 75, 75)
        return img

    def _sharpen(self, img: np.ndarray, strength: float = 0.5) -> np.ndarray:
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]], dtype=np.float32)
        sharpened = cv2.filter2D(img, -1, kernel)
        return cv2.addWeighted(img, 1.0 - strength, sharpened, strength, 0)

    def _background_normalize(self, img: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        bg = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        normalized = cv2.subtract(img, bg)
        return self._normalize(normalized)

    def preprocess(self, img_input, return_steps: bool = False) -> dict:
        t0 = time.time()

        if isinstance(img_input, (str, Path)):
            raw = cv2.imread(str(img_input))
            if raw is None:
                raise ValueError(f"Unable to read image at: {img_input}")
        else:
            raw = img_input.copy()

        quality = self.quality_assessor.assess(raw)
        steps = {"raw": raw.copy()}

        # Grayscale conversion
        img = self._to_gray(raw)
        steps["grayscale"] = img.copy()

        # Contrast normalization
        img = self._normalize(img)
        steps["normalized"] = img.copy()

        # CLAHE local contrast enhancement
        img = self._clahe(img, clip_limit=2.0, tile_grid=(8, 8))
        steps["clahe"] = img.copy()

        # High-frequency noise suppression
        img = self._denoise(img, method="gaussian", kernel=3)
        steps["denoised"] = img.copy()

        # Background gradient normalization
        img = self._background_normalize(img)
        steps["background_norm"] = img.copy()

        # Edge enhancement
        img = self._sharpen(img, strength=0.4)
        steps["sharpened"] = img.copy()

        # Output 3-channel BGR for model input compatibility
        preprocessed = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elapsed_ms = round((time.time() - t0) * 1000, 1)

        result = {
            "raw": raw,
            "preprocessed": preprocessed,
            "quality": quality,
            "elapsed_ms": elapsed_ms,
        }
        if return_steps:
            result["steps"] = steps
        return result
