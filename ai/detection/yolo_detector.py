"""
Sonar object detection interfaces and implementations.
Provides YOLOv8 model inference with a fallback heuristic detector for local development.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional
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
    bbox: list
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    shadow_score: float = 0.0
    evidence_score: float = 0.0
    severity: str = "UNKNOWN"
    area_px: int = 0
    mode: str = "DEMO"
    model_version: str = "best"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InferenceResult:
    image_id: str
    detections: list = field(default_factory=list)
    inference_ms: float = 0.0
    model_mode: str = "DEMO"
    model_version: str = "best"
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


class YOLODetector:
    """Inference engine wrapper around Ultralytics YOLOv8."""

    MODE = "REAL"

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):

        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.version = Path(model_path).stem

    def detect(self, img: np.ndarray, **kwargs) -> list:

        # V3 was trained and validated at 1280 resolution.
        # Keep inference at 1280 so the small-object regime is preserved.
        results = self.model(
            img,
            imgsz=1280,
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

                cls_name = r.names.get(
                    cls_id,
                    "unknown",
                )

                det = Detection(
                    detection_id=f"DET-{uuid.uuid4().hex[:8].upper()}",
                    class_name=cls_name,
                    confidence=round(conf, 3),
                    bbox=[
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ],
                    area_px=(
                        int(x2) - int(x1)
                    ) * (
                        int(y2) - int(y1)
                    ),
                    mode=self.MODE,
                    model_version=self.version,
                )

                detections.append(det)

        return detections


from pathlib import Path
import time
import uuid
from typing import Optional

import numpy as np
import yaml


class SonarDetector:
    """Production YOLO detector. Trained weights are mandatory."""

    def __init__(
        self,
        config_path: str = "configs/model.yaml",
        model_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ):
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

            legacy = Path(
                det_cfg.get(
                    "model_path",
                    "models/detection/yolov8_sonar.pt",
                )
            )

            if primary.exists():
                resolved_path = str(primary)
            elif legacy.exists():
                resolved_path = str(legacy)
            else:
                raise FileNotFoundError(
                    "No trained YOLO model found. "
                    "Expected models/best.pt."
                )

        self.conf_threshold = (
            conf_threshold
            if conf_threshold is not None
            else det_cfg.get(
                "confidence_threshold",
                0.25,
            )
        )

        self.iou_threshold = (
            iou_threshold
            if iou_threshold is not None
            else det_cfg.get(
                "iou_threshold",
                0.45,
            )
        )

        self.model_path = resolved_path

        try:
            self._detector = YOLODetector(
                resolved_path,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load trained YOLO model "
                f"from {resolved_path}: {e}"
            ) from e

        self.mode = "REAL"
        self.model_version = getattr(
            self._detector,
            "version",
            Path(resolved_path).stem,
        )

        print(
            f"[SonarDetector] Loaded YOLO model from {resolved_path}"
        )

        print(
            f"[SonarDetector] Confidence threshold: "
            f"{self.conf_threshold}"
        )

        print(
            f"[SonarDetector] IoU threshold: "
            f"{self.iou_threshold}"
        )

    def run(
        self,
        img: np.ndarray,
        image_id: Optional[str] = None,
        scenario_type: Optional[str] = None,
        annotations: Optional[list] = None,
    ) -> InferenceResult:

        if image_id is None:
            image_id = f"IMG-{uuid.uuid4().hex[:8].upper()}"

        t0 = time.time()

        detections = self._detector.detect(
            img,
            scenario_type=scenario_type,
            annotations=annotations,
        )

        elapsed_ms = round(
            (time.time() - t0) * 1000,
            1,
        )

        return InferenceResult(
            image_id=image_id,
            detections=detections,
            inference_ms=elapsed_ms,
            model_mode=self.mode,
            model_version=self.model_version,
            num_detections=len(detections),
            warning=None,
        )



