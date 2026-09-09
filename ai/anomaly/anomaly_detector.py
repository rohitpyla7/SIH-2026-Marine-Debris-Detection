"""
Production anomaly detection engine for acoustic seabed imagery.

Uses the trained convolutional autoencoder only.
No statistical/demo fallback is used.
"""

import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


@dataclass
class AnomalyResult:
    anomaly_id: str
    anomaly_score: float
    is_anomalous: bool
    bbox: Optional[list]
    confidence: float
    reasons: list
    mode: str = "REAL_AUTOENCODER"

    def to_dict(self) -> dict:
        return asdict(self)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc_enc = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 64 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, 3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, 3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 1, 3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        return self.fc_enc(x)

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(-1, 64, 8, 8)
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))


class AnomalyDetector:
    """
    Production anomaly detector.

    A trained convolutional autoencoder is mandatory.
    """

    MODE = "REAL_AUTOENCODER"

    def __init__(
        self,
        model_path: str = "models/anomaly/autoencoder.pt",
        threshold: float = 0.60,
    ):
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Required anomaly model not found: {path}"
            )

        try:
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            latent_dim = int(checkpoint.get("latent_dim", 32))

            self._ae = ConvAutoencoder(latent_dim)
            self._ae.load_state_dict(checkpoint["state_dict"])
            self._ae.eval()

            self.threshold = float(
                checkpoint.get("threshold", threshold)
            )

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load anomaly autoencoder: {exc}"
            ) from exc

        self.mode = self.MODE

    def _ae_score(self, img: np.ndarray) -> float:
        gray = (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img.ndim == 3
            else img
        )

        resized = cv2.resize(
            gray,
            (64, 64),
            interpolation=cv2.INTER_AREA,
        )

        tensor = torch.from_numpy(
            resized.astype(np.float32) / 255.0
        ).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            reconstructed = self._ae(tensor)

        error = torch.mean(
            (reconstructed - tensor) ** 2
        ).item()

        # Normalize reconstruction error against the trained
        # model threshold while retaining a bounded [0, 1] score.
        score = error / max(self.threshold, 1e-8)

        return float(np.clip(score, 0.0, 1.0))

    def detect_anomalies(
        self,
        img: np.ndarray,
        existing_detections: Optional[list] = None,
        scenario_type: Optional[str] = None,
    ) -> Tuple[List[AnomalyResult], float]:

        t0 = time.time()

        anomaly_score = self._ae_score(img)
        is_anomalous = anomaly_score >= 1.0

        reasons = []

        if is_anomalous:
            reasons.append("High autoencoder reconstruction error")
        else:
            reasons.append("Reconstruction consistent with learned seabed distribution")

        result = AnomalyResult(
            anomaly_id=f"ANO-{uuid.uuid4().hex[:8].upper()}",
            anomaly_score=round(anomaly_score, 3),
            is_anomalous=is_anomalous,
            bbox=None,
            confidence=round(anomaly_score, 3),
            reasons=reasons,
            mode=self.mode,
        )

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        return [result], elapsed_ms
