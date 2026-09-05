"""
Sonar object detection interfaces and implementations.
Provides YOLOv8 model inference with a fallback heuristic detector for local development.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
import uuid
import random
import yaml


KNOWN_CLASSES = [
    "fishing_net",
    "plastic_debris",
    "metal_debris",
    "tire",
    "container",
    "bottle_object",
    "pipe_cable",
    "shipwreck",
    "other_debris",
]


@dataclass
class Detection:
    detection_id: str
    class_name: str
    confidence: float
    bbox: list  # [x1, y1, x2, y2]
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    shadow_score: float = 0.0
    evidence_score: float = 0.0
    severity: str = "UNKNOWN"
    area_px: int = 0
    mode: str = "DEMO"
    model_version: str = "demo-v0.1"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InferenceResult:
    image_id: str
    detections: list = field(default_factory=list)
    inference_ms: float = 0.0
    model_mode: str = "DEMO"
    model_version: str = "demo-v0.1"
    num_detections: int = 0
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "detections": [d.to_dict() for d in self.detections],
            "inference_ms": self.inference_ms,
            "model_mode": self.model_mode,
            "model_version": self.model_version,
            "num_detections": self.num_detections,
            "warning": self.warning,
        }


class DemoDetector:
    """
    Fallback detector used when a trained YOLO model checkpoint is not present.
    Generates synthetic candidate detections matching scenario profiles.
    """

    MODE = "DEMO"
    VERSION = "demo-v0.1"

    SCENARIO_DETECTIONS = {
        "normal_seabed": [],
        "fishing_net": [("fishing_net", 0.87)],
        "metal_debris": [("metal_debris", 0.91), ("metal_debris", 0.78)],
        "plastic_debris": [("plastic_debris", 0.73)],
        "large_structure": [("shipwreck", 0.94)],
        "unknown_anomaly": [],
        "low_quality": [("metal_debris", 0.45)],
        "multi_object": [("fishing_net", 0.85), ("metal_debris", 0.89), ("plastic_debris", 0.68)],
        "strong_shadow": [("metal_debris", 0.93)],
        "tracking_sequence": [("fishing_net", 0.82)],
    }

    def detect(self, img: np.ndarray, scenario_type: Optional[str] = None,
               annotations: Optional[list] = None) -> list:
        detections = []

        if annotations:
            for ann in annotations:
                label = ann.get("label", "unknown_anomaly")
                bbox = ann.get("bbox", [0, 0, 100, 100])

                if label == "unknown_anomaly":
                    continue

                base_conf = self.SCENARIO_DETECTIONS.get(scenario_type or "", [])
                conf = 0.75
                for cls, c in base_conf:
                    if cls == label:
                        conf = c
                        break

                conf = float(np.clip(conf + random.uniform(-0.03, 0.03), 0.3, 0.99))
                x1, y1, x2, y2 = bbox
                area = max(0, (x2 - x1) * (y2 - y1))

                det = Detection(
                    detection_id=f"DET-{uuid.uuid4().hex[:8].upper()}",
                    class_name=label,
                    confidence=round(conf, 3),
                    bbox=bbox,
                    is_anomaly=False,
                    area_px=area,
                    mode=self.MODE,
                    model_version=self.VERSION,
                )
                detections.append(det)

        elif scenario_type in self.SCENARIO_DETECTIONS:
            for label, conf in self.SCENARIO_DETECTIONS[scenario_type]:
                conf = float(np.clip(conf + random.uniform(-0.02, 0.02), 0.3, 0.99))
                h, w = img.shape[:2]
                bbox = [int(w * 0.45), int(h * 0.35), int(w * 0.75), int(h * 0.65)]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                det = Detection(
                    detection_id=f"DET-{uuid.uuid4().hex[:8].upper()}",
                    class_name=label,
                    confidence=round(conf, 3),
                    bbox=bbox,
                    is_anomaly=False,
                    area_px=area,
                    mode=self.MODE,
                    model_version=self.VERSION,
                )
                detections.append(det)

        return detections


class YOLODetector:
    """Inference engine wrapper around Ultralytics YOLOv8."""

    MODE = "REAL"

    def __init__(self, model_path: str, conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45, device: str = "auto"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.version = Path(model_path).stem

    def detect(self, img: np.ndarray, **kwargs) -> list:
        results = self.model(
            img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = r.names.get(cls_id, "unknown")
                det = Detection(
                    detection_id=f"DET-{uuid.uuid4().hex[:8].upper()}",
                    class_name=cls_name,
                    confidence=round(conf, 3),
                    bbox=[int(x1), int(y1), int(x2), int(y2)],
                    area_px=(int(x2) - int(x1)) * (int(y2) - int(y1)),
                    mode=self.MODE,
                    model_version=self.version,
                )
                detections.append(det)
        return detections


class SonarDetector:
    """Top-level detector with automatic YOLO detection and demo fallback."""

    def __init__(self, config_path: str = "configs/model.yaml",
                 model_path: Optional[str] = None):
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            det_cfg = cfg.get("detection", {})
        except Exception:
            det_cfg = {}

        if model_path is not None:
            resolved_path = model_path
        else:
            primary = Path("models/best.pt")
            legacy = Path(det_cfg.get("model_path", "models/detection/yolov8_sonar.pt"))
            resolved_path = str(primary) if primary.exists() else str(legacy)

        self.conf_threshold = det_cfg.get("confidence_threshold", 0.25)
        self.iou_threshold = det_cfg.get("iou_threshold", 0.45)
        self.model_path = resolved_path

        if Path(resolved_path).exists():
            try:
                self._detector = YOLODetector(
                    resolved_path,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold,
                )
                self.mode = "REAL"
                print(f"[SonarDetector] Loaded YOLO model from {resolved_path}")
            except Exception as e:
                print(f"[SonarDetector] Failed to load YOLO ({e}), reverting to fallback")
                self._detector = DemoDetector()
                self.mode = "DEMO"
        else:
            self._detector = DemoDetector()
            self.mode = "DEMO"

    def run(self, img: np.ndarray, image_id: Optional[str] = None,
            scenario_type: Optional[str] = None,
            annotations: Optional[list] = None) -> InferenceResult:
        if image_id is None:
            image_id = f"IMG-{uuid.uuid4().hex[:8].upper()}"

        t0 = time.time()
        detections = self._detector.detect(
            img,
            scenario_type=scenario_type,
            annotations=annotations,
        )
        elapsed_ms = round((time.time() - t0) * 1000, 1)

        warning = None
        if self.mode == "DEMO":
            warning = "Running in demo mode. Place trained weights at models/best.pt for live inference."

        return InferenceResult(
            image_id=image_id,
            detections=detections,
            inference_ms=elapsed_ms,
            model_mode=self.mode,
            model_version=getattr(self._detector, "version", "demo-v0.1"),
            num_detections=len(detections),
            warning=warning,
        )
