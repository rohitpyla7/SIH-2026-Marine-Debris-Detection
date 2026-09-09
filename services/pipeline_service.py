"""
End-to-end pipeline service for sonar imagery ingestion, enhancement,
detection, anomaly analysis, shadow analysis, and evidence fusion.
"""

import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.preprocessing.sonar_preprocessor import SonarPreprocessor
from ai.detection.yolo_detector import SonarDetector, Detection, InferenceResult
from ai.anomaly.anomaly_detector import AnomalyDetector
from ai.shadow_analysis.shadow_analyzer import AcousticShadowAnalyzer
from ai.fusion.confidence_fusion import (
    compute_evidence_score,
    compute_anomaly_evidence_score,
    compute_severity,
    get_severity_color,
)


@dataclass
class PipelineDetection:
    detection_id: str
    class_name: str
    display_name: str
    confidence: float
    anomaly_score: float
    shadow_score: float
    evidence_score: float
    evidence_pct: float
    severity: str
    severity_color: str
    bbox: list
    bbox_width_px: int
    bbox_height_px: int
    object_area_px: int
    is_anomaly: bool
    shadow_details: dict = field(default_factory=dict)
    fusion_breakdown: dict = field(default_factory=dict)
    mode: str = "DEMO"
    model_version: str = "best"
    track_id: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    depth_m: Optional[float] = None
    geo_label: str = "Simulated Location"
    height_estimate: str = "Unavailable"
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "detection_id": self.detection_id,
            "class_name": self.class_name,
            "display_name": self.display_name,
            "confidence": self.confidence,
            "confidence_pct": round(self.confidence * 100, 1),
            "anomaly_score": self.anomaly_score,
            "shadow_score": self.shadow_score,
            "evidence_score": self.evidence_score,
            "evidence_pct": self.evidence_pct,
            "severity": self.severity,
            "severity_color": self.severity_color,
            "bbox": self.bbox,
            "bbox_width_px": self.bbox_width_px,
            "bbox_height_px": self.bbox_height_px,
            "object_area_px": self.object_area_px,
            "is_anomaly": self.is_anomaly,
            "mode": self.mode,
            "model_version": self.model_version,
            "track_id": self.track_id,
            "lat": self.lat,
            "lon": self.lon,
            "depth_m": self.depth_m,
            "geo_label": self.geo_label,
            "height_estimate": self.height_estimate,
            "reasons": self.reasons,
        }


@dataclass
class PipelineResult:
    image_id: str
    raw_image: np.ndarray
    preprocessed_image: np.ndarray
    annotated_image: np.ndarray
    quality: dict
    detections: list
    anomaly_result: dict
    timing: dict
    mode: str
    model_version: str
    warnings: list
    num_known: int = 0
    num_anomalies: int = 0
    natural_filter_applied: bool = False

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "quality": self.quality,
            "detections": [d.to_dict() for d in self.detections],
            "anomaly_result": self.anomaly_result,
            "timing": self.timing,
            "mode": self.mode,
            "model_version": self.model_version,
            "warnings": self.warnings,
            "num_known": self.num_known,
            "num_anomalies": self.num_anomalies,
        }


NATURAL_CLASSES = {
    "rock",
    "natural_formation",
    "sand_ripple",
    "seabed_feature",
}

ARTIFICIAL_CLASSES = {
    "fishing_net",
    "plastic_debris",
    "metal_debris",
    "tire",
    "container",
    "bottle_object",
    "pipe_cable",
    "shipwreck",
    "other_debris",
    "pipeline",
    "unknown_anomaly",
}


def classify_natural_vs_artificial(
    class_name: str,
    confidence: float,
    anomaly_score: float,
) -> dict:
    cn = class_name.lower().replace(" ", "_")

    if cn in NATURAL_CLASSES:
        return {
            "type": "NATURAL",
            "is_manmade": False,
            "reason": "Natural seabed feature morphology",
        }

    if cn in ARTIFICIAL_CLASSES:
        return {
            "type": "ARTIFICIAL",
            "is_manmade": True,
            "reason": "Anthropogenic debris target profile",
        }

    if anomaly_score > 0.65:
        return {
            "type": "ARTIFICIAL_UNKNOWN",
            "is_manmade": True,
            "reason": "High anomaly score suggests non-natural morphology",
        }

    return {
        "type": "UNCERTAIN",
        "is_manmade": None,
        "reason": "Indeterminate morphological structure",
    }


SEVERITY_COLORS_BGR = {
    "HIGH": (50, 50, 220),
    "MEDIUM": (50, 160, 255),
    "LOW": (50, 200, 80),
    "UNKNOWN": (150, 150, 150),
}


CLASS_DISPLAY_NAMES = {
    "fishing_net": "Fishing Net",
    "plastic_debris": "Plastic Debris",
    "metal_debris": "Metal Debris",
    "tire": "Tire",
    "container": "Container",
    "bottle_object": "Bottle/Object",
    "pipe_cable": "Pipe/Cable",
    "shipwreck": "Shipwreck",
    "other_debris": "Other Debris",
    "pipeline": "Pipeline",
    "unknown_anomaly": "Unknown Anomaly",
    "rock": "Rock (Natural)",
    "natural_formation": "Natural Formation",
}


def annotate_image(
    img: np.ndarray,
    detections: list,
    anomaly_result: dict,
    quality: dict,
    mode: str = "DEMO",
) -> np.ndarray:

    vis = img.copy()

    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    h, w = vis.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]

        color = SEVERITY_COLORS_BGR.get(
            det.severity,
            (150, 150, 150),
        )

        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        label = f"{det.display_name} {det.confidence:.0%}"

        if det.is_anomaly:
            label = f"ANOMALY {det.anomaly_score:.2f}"

        evidence_label = (
            f"E:{det.evidence_pct:.0f}% {det.severity}"
        )

        label_y = max(y1 - 22, 16)

        (lw, lh), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            1,
        )

        cv2.rectangle(
            vis,
            (x1, label_y - lh - 4),
            (x1 + lw + 4, label_y + 2),
            color,
            -1,
        )

        cv2.putText(
            vis,
            label,
            (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        ev_y = label_y + 14

        (ew, eh), _ = cv2.getTextSize(
            evidence_label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            1,
        )

        cv2.rectangle(
            vis,
            (x1, ev_y - eh - 2),
            (x1 + ew + 4, ev_y + 2),
            (30, 30, 30),
            -1,
        )

        cv2.putText(
            vis,
            evidence_label,
            (x1 + 2, ev_y - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (220, 220, 220),
            1,
        )

        if det.shadow_details.get("shadow_bbox"):
            sx1, sy1, sx2, sy2 = [
                int(v)
                for v in det.shadow_details["shadow_bbox"]
            ]

            overlay = vis.copy()

            cv2.rectangle(
                overlay,
                (sx1, sy1),
                (sx2, sy2),
                (30, 100, 200),
                -1,
            )

            cv2.addWeighted(
                overlay,
                0.25,
                vis,
                0.75,
                0,
                vis,
            )

            cv2.rectangle(
                vis,
                (sx1, sy1),
                (sx2, sy2),
                (30, 80, 180),
                1,
            )

            cv2.putText(
                vis,
                "SHADOW",
                (sx1 + 2, max(sy1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (100, 160, 255),
                1,
            )

    q_label = quality.get(
        "quality",
        "UNKNOWN",
    )

    q_color = {
        "GOOD": (50, 200, 80),
        "ACCEPTABLE": (50, 160, 255),
        "POOR": (50, 50, 220),
        "UNKNOWN": (150, 150, 150),
    }.get(
        q_label,
        (150, 150, 150),
    )

    cv2.putText(
        vis,
        f"Quality: {q_label}",
        (8, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        q_color,
        1,
    )

    if mode == "REAL":
        wm_text = "YOLO ACTIVE"
        wm_color = (50, 200, 80)
    else:
        wm_text = "DEMO"
        wm_color = (120, 120, 120)

    (ww, _), _ = cv2.getTextSize(
        wm_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        1,
    )

    cv2.putText(
        vis,
        wm_text,
        (w - ww - 6, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        wm_color,
        1,
    )

    return vis


class SonarAnalysisPipeline:
    """Orchestrates image ingestion, preprocessing, detection, and fusion."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        self.preprocessor = SonarPreprocessor()

        self.detector = SonarDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        self.anomaly_detector = AnomalyDetector()
        self.shadow_analyzer = AcousticShadowAnalyzer()

    def run(
        self,
        img_input,
        image_id: Optional[str] = None,
        scenario_type: Optional[str] = None,
        annotations: Optional[list] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        depth_m: Optional[float] = None,
        sonar_metadata: Optional[dict] = None,
    ) -> PipelineResult:

        if image_id is None:
            image_id = (
                f"IMG-{uuid.uuid4().hex[:8].upper()}"
            )

        timing = {}
        warnings = []

        t_total = time.time()

        # ---------------------------------------------------------
        # LOAD IMAGE
        # ---------------------------------------------------------

        if isinstance(img_input, (str, Path)):
            raw = cv2.imread(str(img_input))

            if raw is None:
                raise ValueError(
                    f"Unable to read image at: {img_input}"
                )

        else:
            raw = img_input.copy()

            if raw.ndim == 2:
                raw = cv2.cvtColor(
                    raw,
                    cv2.COLOR_GRAY2BGR,
                )

        # ---------------------------------------------------------
        # PREPROCESSING
        # ---------------------------------------------------------

        t0 = time.time()

        try:
            prep_result = self.preprocessor.preprocess(
                raw,
                return_steps=True,
            )

            preprocessed = prep_result["preprocessed"]
            quality = prep_result["quality"]

            timing["preprocess_ms"] = prep_result[
                "elapsed_ms"
            ]

        except Exception as e:
            warnings.append(
                f"Preprocessing warning: {e}"
            )

            preprocessed = raw.copy()

            quality = {
                "quality": "UNKNOWN",
                "score": 0.5,
                "reasons": [],
                "metrics": {},
            }

            timing["preprocess_ms"] = round(
                (time.time() - t0) * 1000,
                1,
            )

        if quality["quality"] == "POOR":
            warnings.append(
                "Low image quality detected; confidence estimates adjusted."
            )

        # ---------------------------------------------------------
        # YOLO DETECTION
        # IMPORTANT: raw sonar image is sent to YOLO.
        # ---------------------------------------------------------

        t0 = time.time()

        try:
            det_result = self.detector.run(
                raw,
                image_id=image_id,
                scenario_type=scenario_type,
                annotations=annotations,
            )

            raw_detections = det_result.detections

            timing["detection_ms"] = (
                det_result.inference_ms
            )

            if det_result.warning:
                warnings.append(
                    det_result.warning
                )

        except Exception as e:
            warnings.append(
                f"Detection error: {e}"
            )

            raw_detections = []

            timing["detection_ms"] = round(
                (time.time() - t0) * 1000,
                1,
            )

        # ---------------------------------------------------------
        # ANOMALY DETECTION
        # ---------------------------------------------------------

        t0 = time.time()

        try:
            anomaly_results, anom_ms = (
                self.anomaly_detector.detect_anomalies(
                    preprocessed,
                    existing_detections=raw_detections,
                    scenario_type=scenario_type,
                )
            )

            timing["anomaly_ms"] = anom_ms

            anom = (
                anomaly_results[0]
                if anomaly_results
                else None
            )

        except Exception as e:
            warnings.append(
                f"Anomaly detection error: {e}"
            )

            anom = None

            timing["anomaly_ms"] = round(
                (time.time() - t0) * 1000,
                1,
            )

        anomaly_summary = (
            anom.to_dict()
            if anom
            else {
                "anomaly_score": 0.0,
                "is_anomalous": False,
                "reasons": [],
                "mode": "UNAVAILABLE",
            }
        )

        # ---------------------------------------------------------
        # SHADOW ANALYSIS + EVIDENCE FUSION
        # ---------------------------------------------------------

        t_shadow = 0.0

        pipeline_detections = []

        for det in raw_detections:

            try:
                shadow_r = self.shadow_analyzer.analyze(
                    preprocessed,
                    det.bbox,
                    sonar_metadata,
                )

                shadow_score = shadow_r.shadow_score

                shadow_details = shadow_r.to_dict()

                t_shadow += shadow_r.elapsed_ms

            except Exception as e:
                shadow_score = 0.0

                shadow_details = {
                    "error": str(e)
                }

            iq_score = quality.get(
                "score",
                0.8,
            )

            anom_score_for_known = (
                anomaly_summary.get(
                    "anomaly_score",
                    0.0,
                )
            )

            fusion = compute_evidence_score(
                detector_confidence=det.confidence,
                shadow_score=shadow_score,
                anomaly_score=anom_score_for_known,
                image_quality_score=iq_score,
                texture_score=0.6,
                shape_score=0.6,
            )

            severity = compute_severity(
                class_name=det.class_name,
                evidence_score=fusion[
                    "evidence_score"
                ],
                is_anomaly=False,
                object_area_px=det.area_px,
            )

            nat_filter = (
                classify_natural_vs_artificial(
                    det.class_name,
                    det.confidence,
                    anom_score_for_known,
                )
            )

            x1, y1, x2, y2 = det.bbox

            pd = PipelineDetection(
                detection_id=det.detection_id,
                class_name=det.class_name,
                display_name=CLASS_DISPLAY_NAMES.get(
                    det.class_name,
                    det.class_name.replace(
                        "_",
                        " ",
                    ).title(),
                ),
                confidence=det.confidence,
                anomaly_score=anom_score_for_known,
                shadow_score=shadow_score,
                evidence_score=fusion[
                    "evidence_score"
                ],
                evidence_pct=fusion[
                    "evidence_pct"
                ],
                severity=severity,
                severity_color=get_severity_color(
                    severity
                ),
                bbox=det.bbox,
                bbox_width_px=max(
                    0,
                    x2 - x1,
                ),
                bbox_height_px=max(
                    0,
                    y2 - y1,
                ),
                object_area_px=det.area_px,
                is_anomaly=False,
                shadow_details=shadow_details,
                fusion_breakdown=fusion[
                    "breakdown"
                ],
                mode=det.mode,
                model_version=det.model_version,
                lat=lat,
                lon=lon,
                depth_m=depth_m,
                geo_label=(
                    "Simulated Coordinates"
                    if lat is not None
                    else "Location Unavailable"
                ),
                height_estimate=shadow_details.get(
                    "height_estimate",
                    "Unavailable",
                ),
                reasons=(
                    [f"Type: {nat_filter['type']}"]
                    + anomaly_summary.get(
                        "reasons",
                        [],
                    )[:2]
                ),
            )

            pipeline_detections.append(pd)

        timing["shadow_ms"] = round(
            t_shadow,
            1,
        )

        # ---------------------------------------------------------
        # UNKNOWN ANOMALY RESULT
        # ---------------------------------------------------------

        if (
            anom
            and anom.is_anomalous
            and anom.bbox
        ):

            fusion_anom = (
                compute_anomaly_evidence_score(
                    anomaly_score=anom.anomaly_score,
                    shadow_score=0.3,
                    image_quality_score=quality.get(
                        "score",
                        0.8,
                    ),
                )
            )

            sev = compute_severity(
                class_name="unknown_anomaly",
                evidence_score=fusion_anom[
                    "evidence_score"
                ],
                is_anomaly=True,
                anomaly_score=anom.anomaly_score,
            )

            ax1, ay1, ax2, ay2 = anom.bbox

            anom_pd = PipelineDetection(
                detection_id=anom.anomaly_id,
                class_name="unknown_anomaly",
                display_name="Unknown Anomaly",
                confidence=anom.confidence,
                anomaly_score=anom.anomaly_score,
                shadow_score=0.0,
                evidence_score=fusion_anom[
                    "evidence_score"
                ],
                evidence_pct=fusion_anom[
                    "evidence_pct"
                ],
                severity=sev,
                severity_color=get_severity_color(
                    sev
                ),
                bbox=anom.bbox,
                bbox_width_px=max(
                    0,
                    ax2 - ax1,
                ),
                bbox_height_px=max(
                    0,
                    ay2 - ay1,
                ),
                object_area_px=max(
                    0,
                    (ax2 - ax1)
                    * (ay2 - ay1),
                ),
                is_anomaly=True,
                shadow_details={},
                fusion_breakdown=fusion_anom.get(
                    "breakdown",
                    {},
                ),
                mode=anom.mode,
                model_version="autoencoder-v1",
                lat=lat,
                lon=lon,
                depth_m=depth_m,
                geo_label=(
                    "Simulated Coordinates"
                    if lat is not None
                    else "Location Unavailable"
                ),
                height_estimate="Unavailable",
                reasons=anom.reasons,
            )

            pipeline_detections.append(
                anom_pd
            )

        # ---------------------------------------------------------
        # ANNOTATED IMAGE
        # ---------------------------------------------------------

        annotated = annotate_image(
            preprocessed,
            pipeline_detections,
            anomaly_summary,
            quality,
            mode=self.detector.mode,
        )

        timing["total_ms"] = round(
            (time.time() - t_total) * 1000,
            1,
        )

        num_known = sum(
            1
            for d in pipeline_detections
            if not d.is_anomaly
        )

        num_anomalies = sum(
            1
            for d in pipeline_detections
            if d.is_anomaly
        )

        return PipelineResult(
            image_id=image_id,
            raw_image=raw,
            preprocessed_image=preprocessed,
            annotated_image=annotated,
            quality=quality,
            detections=pipeline_detections,
            anomaly_result=anomaly_summary,
            timing=timing,
            mode=self.detector.mode,
            model_version=getattr(
                self.detector._detector,
                "version",
                "best",
            ),
            warnings=warnings,
            num_known=num_known,
            num_anomalies=num_anomalies,
        )


_pipeline_instance = None


def get_pipeline() -> SonarAnalysisPipeline:
    global _pipeline_instance

    if _pipeline_instance is None:
        _pipeline_instance = SonarAnalysisPipeline()

    return _pipeline_instance

