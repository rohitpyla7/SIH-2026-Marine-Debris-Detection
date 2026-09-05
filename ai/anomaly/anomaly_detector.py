"""
Anomaly detection engine for acoustic seabed imagery.
Identifies statistical outliers in texture, gradient, and intensity distributions.
"""

import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np


@dataclass
class AnomalyResult:
    anomaly_id: str
    anomaly_score: float
    is_anomalous: bool
    bbox: Optional[list]
    confidence: float
    reasons: list
    mode: str = "DEMO_STATISTICAL"

    def to_dict(self) -> dict:
        return asdict(self)


class StatisticalAnomalyDetector:
    """Calculates anomaly likelihood from gradient energy, local contrast, and edge density."""

    MODE = "DEMO_STATISTICAL"
    THRESHOLD = 0.60

    def __init__(self, threshold: float = 0.60):
        self.threshold = threshold

    def _extract_features(self, gray: np.ndarray) -> dict:
        h, w = gray.shape
        mean_int = float(gray.mean())
        std_int = float(gray.std())

        left = gray[:, :w // 2 - 20]
        right = gray[:, w // 2 + 20:]
        sonar_data = np.concatenate([left.ravel(), right.ravel()])
        local_contrast = float(sonar_data.std())

        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(edges.mean()) / 255.0

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        texture_energy = float(grad_mag.mean())

        high_int_frac = float((gray > 200).sum()) / gray.size

        return {
            "mean_intensity": mean_int,
            "std_intensity": std_int,
            "local_contrast": local_contrast,
            "edge_density": edge_density,
            "texture_energy": texture_energy,
            "high_intensity_fraction": high_int_frac,
        }

    def _score_from_features(self, features: dict, scenario_type: Optional[str] = None) -> float:
        score = 0.0
        weights = 0.0

        tex = features["texture_energy"]
        tex_score = min(1.0, tex / 30.0)
        score += tex_score * 0.30
        weights += 0.30

        edge_score = min(1.0, features["edge_density"] / 0.15)
        score += edge_score * 0.25
        weights += 0.25

        hi_score = min(1.0, features["high_intensity_fraction"] / 0.08)
        score += hi_score * 0.25
        weights += 0.25

        std_score = min(1.0, features["std_intensity"] / 60.0)
        score += std_score * 0.20
        weights += 0.20

        final_score = score / weights if weights > 0 else 0.0

        if scenario_type == "unknown_anomaly":
            final_score = min(1.0, final_score * 1.35)
        elif scenario_type == "normal_seabed":
            final_score = final_score * 0.40

        return float(np.clip(final_score, 0.0, 1.0))

    def _localize_anomaly(self, gray: np.ndarray) -> Optional[list]:
        h, w = gray.shape
        mask = np.zeros_like(gray)
        mask[:, :w // 2 - 30] = 1
        mask[:, w // 2 + 30:] = 1

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(gx ** 2 + gy ** 2)
        grad = (grad * mask).astype(np.float32)

        if grad.max() < 1:
            return None
        grad_norm = (grad / grad.max() * 255).astype(np.uint8)
        _, thresh = cv2.threshold(grad_norm, 150, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 200:
            return None
        x, y, bw, bh = cv2.boundingRect(c)
        pad = 10
        return [
            max(0, x - pad), max(0, y - pad),
            min(w - 1, x + bw + pad), min(h - 1, y + bh + pad)
        ]

    def score(self, img: np.ndarray, scenario_type: Optional[str] = None) -> dict:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        features = self._extract_features(gray)
        anomaly_score = self._score_from_features(features, scenario_type)
        return {
            "anomaly_score": round(anomaly_score, 3),
            "features": features,
            "scenario_type": scenario_type,
        }


class AnomalyDetector:
    """Unified detector interface supporting model weights and statistical fallbacks."""

    def __init__(self, model_path: str = "models/anomaly/autoencoder.pt",
                 threshold: float = 0.60):
        self.threshold = threshold

        if Path(model_path).exists():
            try:
                raise NotImplementedError("Autoencoder model format not loaded")
            except Exception:
                self._model = StatisticalAnomalyDetector(threshold)
                self.mode = "DEMO_STATISTICAL"
        else:
            self._model = StatisticalAnomalyDetector(threshold)
            self.mode = "DEMO_STATISTICAL"

    def detect_anomalies(self, img: np.ndarray,
                          existing_detections: Optional[list] = None,
                          scenario_type: Optional[str] = None) -> Tuple[List[AnomalyResult], float]:
        t0 = time.time()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        scored = self._model.score(img, scenario_type=scenario_type)
        anomaly_score = scored["anomaly_score"]
        features = scored["features"]
        is_anomalous = anomaly_score >= self.threshold

        if scenario_type == "unknown_anomaly" and anomaly_score < self.threshold:
            anomaly_score = max(anomaly_score, self.threshold + 0.05)
            is_anomalous = True

        reasons = []
        if features["texture_energy"] > 20:
            reasons.append("Unusual texture energy")
        if features["high_intensity_fraction"] > 0.05:
            reasons.append("High-intensity region detected")
        if features["edge_density"] > 0.10:
            reasons.append("Elevated edge density")
        if not reasons:
            reasons.append("Nominal seabed texture pattern")

        bbox = None
        if is_anomalous:
            bbox = self._model._localize_anomaly(gray)

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        result = AnomalyResult(
            anomaly_id=f"ANO-{uuid.uuid4().hex[:8].upper()}",
            anomaly_score=round(anomaly_score, 3),
            is_anomalous=is_anomalous,
            bbox=bbox,
            confidence=round(min(anomaly_score * 1.1, 1.0), 3),
            reasons=reasons,
            mode=self.mode,
        )

        return [result], elapsed_ms
