"""
Database and persistence operations for missions, acoustic frames, and target detections.
"""

import csv
import io
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.models import (
    get_session, Mission, SonarImage, Detection, Track, init_db
)

DB_PATH = "sih26057.db"


def _session():
    return get_session(DB_PATH)


def init_database():
    init_db(DB_PATH)


def create_mission(mission_id: str, name: str, date: str = None,
                   area: str = None, operator: str = None,
                   data_label: str = "SURVEY") -> dict:
    db = _session()
    try:
        existing = db.query(Mission).filter_by(mission_id=mission_id).first()
        if existing:
            return {"status": "exists", "mission_id": mission_id}
        m = Mission(
            mission_id=mission_id,
            name=name,
            date=date or datetime.utcnow().strftime("%Y-%m-%d"),
            area=area or "Survey Grid Alpha",
            operator=operator or "Autonomous Controller",
            data_label=data_label,
            status="completed",
        )
        db.add(m)
        db.commit()
        return {"status": "created", "mission_id": mission_id}
    finally:
        db.close()


def get_all_missions() -> list:
    db = _session()
    try:
        ms = db.query(Mission).order_by(Mission.created_at.desc()).all()
        result = []
        for m in ms:
            img_count = db.query(SonarImage).filter_by(mission_id=m.mission_id).count()
            det_count = db.query(Detection).filter_by(mission_id=m.mission_id).count()
            anom_count = db.query(Detection).filter_by(
                mission_id=m.mission_id, is_anomaly=True).count()
            result.append({
                "mission_id": m.mission_id,
                "name": m.name,
                "date": m.date,
                "area": m.area,
                "status": m.status,
                "data_label": m.data_label,
                "image_count": img_count,
                "detection_count": det_count,
                "anomaly_count": anom_count,
            })
        return result
    finally:
        db.close()


def get_mission(mission_id: str) -> Optional[dict]:
    db = _session()
    try:
        m = db.query(Mission).filter_by(mission_id=mission_id).first()
        if not m:
            return None
        images = db.query(SonarImage).filter_by(mission_id=mission_id).all()
        dets = db.query(Detection).filter_by(mission_id=mission_id).all()
        return {
            "mission_id": m.mission_id,
            "name": m.name,
            "date": m.date,
            "area": m.area,
            "status": m.status,
            "data_label": m.data_label,
            "image_count": len(images),
            "detection_count": len(dets),
            "anomaly_count": sum(1 for d in dets if d.is_anomaly),
            "high_risk_count": sum(1 for d in dets if d.severity == "HIGH"),
        }
    finally:
        db.close()


def save_image_record(mission_id: str, image_id: str, filename: str,
                      filepath: str, scenario_type: str = None,
                      description: str = None, quality_label: str = "UNKNOWN",
                      quality_score: float = 0.0, lat: float = None,
                      lon: float = None, depth_m: float = None,
                      data_label: str = "SURVEY") -> dict:
    db = _session()
    try:
        existing = db.query(SonarImage).filter_by(image_id=image_id).first()
        if existing:
            return {"status": "exists", "image_id": image_id}
        img = SonarImage(
            image_id=image_id,
            mission_id=mission_id,
            filename=filename,
            filepath=filepath,
            scenario_type=scenario_type,
            description=description,
            quality_label=quality_label,
            quality_score=quality_score,
            lat=lat,
            lon=lon,
            depth_m=depth_m,
            data_label=data_label,
        )
        db.add(img)
        db.commit()
        return {"status": "created", "image_id": image_id}
    finally:
        db.close()


def get_mission_images(mission_id: str) -> list:
    db = _session()
    try:
        imgs = db.query(SonarImage).filter_by(mission_id=mission_id).order_by(
            SonarImage.image_id).all()
        return [{
            "image_id": i.image_id,
            "filename": i.filename,
            "filepath": i.filepath,
            "scenario_type": i.scenario_type,
            "description": i.description,
            "quality_label": i.quality_label,
            "quality_score": i.quality_score,
            "lat": i.lat,
            "lon": i.lon,
            "depth_m": i.depth_m,
            "data_label": i.data_label,
            "processed": i.processed,
        } for i in imgs]
    finally:
        db.close()


def mark_image_processed(image_id: str, result_path: str = None):
    db = _session()
    try:
        img = db.query(SonarImage).filter_by(image_id=image_id).first()
        if img:
            img.processed = True
            if result_path:
                img.result_path = result_path
            db.commit()
    finally:
        db.close()


def save_detection(mission_id: str, image_id: str, det,
                   lat: float = None, lon: float = None,
                   depth_m: float = None) -> str:
    db = _session()
    try:
        x1, y1, x2, y2 = det.bbox if det.bbox else [0, 0, 0, 0]
        d = Detection(
            detection_id=det.detection_id,
            image_id=image_id,
            mission_id=mission_id,
            class_name=det.class_name,
            confidence=det.confidence,
            is_anomaly=det.is_anomaly,
            anomaly_score=det.anomaly_score,
            shadow_score=det.shadow_score,
            evidence_score=det.evidence_score,
            severity=det.severity,
            bbox_x1=int(x1), bbox_y1=int(y1),
            bbox_x2=int(x2), bbox_y2=int(y2),
            object_area=det.object_area_px,
            lat=lat or det.lat,
            lon=lon or det.lon,
            depth_m=depth_m or det.depth_m,
            mode=det.mode,
            model_version=det.model_version,
            data_label="SURVEY",
        )
        db.add(d)
        db.commit()
        return det.detection_id
    except Exception as e:
        db.rollback()
        print(f"[DB] Save detection error: {e}")
        return det.detection_id
    finally:
        db.close()


def get_all_detections(mission_id: str = None, limit: int = 200) -> list:
    db = _session()
    try:
        q = db.query(Detection)
        if mission_id:
            q = q.filter_by(mission_id=mission_id)
        dets = q.order_by(Detection.created_at.desc()).limit(limit).all()
        return [{
            "detection_id": d.detection_id,
            "image_id": d.image_id,
            "mission_id": d.mission_id,
            "class_name": d.class_name,
            "confidence": d.confidence,
            "is_anomaly": d.is_anomaly,
            "anomaly_score": d.anomaly_score,
            "shadow_score": d.shadow_score,
            "evidence_score": d.evidence_score,
            "severity": d.severity,
            "bbox": [d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2],
            "lat": d.lat, "lon": d.lon, "depth_m": d.depth_m,
            "mode": d.mode, "model_version": d.model_version,
            "operator_status": d.operator_status,
            "operator_label": d.operator_label,
            "operator_note": d.operator_note,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        } for d in dets]
    finally:
        db.close()


def update_operator_feedback(detection_id: str, status: str,
                              label: str = None, note: str = None):
    db = _session()
    try:
        d = db.query(Detection).filter_by(detection_id=detection_id).first()
        if d:
            d.operator_status = status
            d.operator_label = label
            d.operator_note = note
            d.reviewed_at = datetime.utcnow()
            db.commit()
            return True
        return False
    finally:
        db.close()


def get_statistics() -> dict:
    db = _session()
    try:
        total_missions = db.query(Mission).count()
        total_images = db.query(SonarImage).count()
        total_detections = db.query(Detection).count()
        total_anomalies = db.query(Detection).filter_by(is_anomaly=True).count()
        high_risk = db.query(Detection).filter_by(severity="HIGH").count()

        all_dets = db.query(Detection).all()
        avg_evidence = (sum(d.evidence_score for d in all_dets) / len(all_dets)
                        if all_dets else 0.0)

        class_dist = {}
        for d in all_dets:
            cn = d.class_name or "unknown"
            class_dist[cn] = class_dist.get(cn, 0) + 1

        sev_dist = {}
        for d in all_dets:
            s = d.severity or "UNKNOWN"
            sev_dist[s] = sev_dist.get(s, 0) + 1

        return {
            "total_missions": total_missions,
            "total_images": total_images,
            "total_detections": total_detections,
            "total_anomalies": total_anomalies,
            "known_debris": total_detections - total_anomalies,
            "high_risk": high_risk,
            "avg_evidence_score": round(avg_evidence, 3),
            "avg_evidence_pct": round(avg_evidence * 100, 1),
            "class_distribution": class_dist,
            "severity_distribution": sev_dist,
        }
    finally:
        db.close()


def export_detections_csv(mission_id: str = None) -> str:
    dets = get_all_detections(mission_id=mission_id, limit=10000)
    if not dets:
        return "No detections found."
    fields = [
        "detection_id", "mission_id", "image_id", "class_name",
        "confidence", "anomaly_score", "shadow_score", "evidence_score",
        "severity", "is_anomaly", "lat", "lon", "depth_m",
        "mode", "operator_status", "created_at"
    ]
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(dets)
    return out.getvalue()


def export_detections_json(mission_id: str = None) -> str:
    dets = get_all_detections(mission_id=mission_id, limit=10000)
    return json.dumps({
        "export_timestamp": datetime.utcnow().isoformat(),
        "mission_id": mission_id or "ALL",
        "total_detections": len(dets),
        "detections": dets,
    }, indent=2, default=str)
