"""
Multi-frame object tracking across sequential sonar records.
Uses spatial Intersection over Union (IoU) and centroid proximity for frame-to-frame association.
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np


def iou(box_a: list, box_b: list) -> float:
    """Compute Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def centroid_dist(box_a: list, box_b: list) -> float:
    """Calculate Euclidean distance between bounding box centers."""
    cx_a = (box_a[0] + box_a[2]) / 2.0
    cy_a = (box_a[1] + box_a[3]) / 2.0
    cx_b = (box_b[0] + box_b[2]) / 2.0
    cy_b = (box_b[1] + box_b[3]) / 2.0
    return float(np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2))


@dataclass
class Track:
    track_id: str
    class_name: str
    bbox: list
    frame_id: int
    detection_ids: list = field(default_factory=list)
    frames_seen: int = 1
    frames_lost: int = 0
    status: str = "active"
    mission_id: Optional[str] = None

    @property
    def is_confirmed(self) -> bool:
        return self.frames_seen >= 2

    def update(self, bbox: list, detection_id: str, frame_id: int):
        self.bbox = bbox
        self.detection_ids.append(detection_id)
        self.frames_seen += 1
        self.frames_lost = 0
        self.frame_id = frame_id
        if self.frames_seen >= 2:
            self.status = "confirmed"


class SimpleTracker:
    """Tracks detected objects across sequential frames to reduce duplicate alerts."""

    def __init__(self, iou_threshold: float = 0.3, max_frames_lost: int = 5,
                 max_centroid_dist: float = 80.0):
        self.iou_threshold = iou_threshold
        self.max_frames_lost = max_frames_lost
        self.max_centroid_dist = max_centroid_dist
        self.tracks: Dict[str, Track] = {}
        self.frame_count = 0

    def update(self, detections: list, frame_id: Optional[int] = None,
               mission_id: Optional[str] = None) -> list:
        if frame_id is None:
            frame_id = self.frame_count
        self.frame_count += 1

        active_tracks = [t for t in self.tracks.values() if t.status != "lost"]
        matched_tracks = set()
        matched_dets = set()
        results = []

        for det in detections:
            best_track = None
            best_score = -1.0

            for track in active_tracks:
                if track.track_id in matched_tracks or track.class_name != det.class_name:
                    continue

                score = iou(det.bbox, track.bbox)
                if score < self.iou_threshold:
                    dist = centroid_dist(det.bbox, track.bbox)
                    if dist < self.max_centroid_dist:
                        score = max(score, 1.0 - dist / self.max_centroid_dist)

                if score > best_score:
                    best_score = score
                    best_track = track

            if best_track is not None and best_score >= self.iou_threshold:
                best_track.update(det.bbox, det.detection_id, frame_id)
                matched_tracks.add(best_track.track_id)
                matched_dets.add(id(det))
                results.append((det, best_track.track_id))
            else:
                new_tid = f"TRK-{uuid.uuid4().hex[:6].upper()}"
                new_track = Track(
                    track_id=new_tid,
                    class_name=det.class_name,
                    bbox=det.bbox,
                    frame_id=frame_id,
                    detection_ids=[det.detection_id],
                    mission_id=mission_id,
                )
                self.tracks[new_tid] = new_track
                results.append((det, new_tid))

        for track in active_tracks:
            if track.track_id not in matched_tracks:
                track.frames_lost += 1
                if track.frames_lost >= self.max_frames_lost:
                    track.status = "lost"

        return results

    def get_active_tracks(self) -> list:
        return [t for t in self.tracks.values() if t.status != "lost"]

    def get_confirmed_tracks(self) -> list:
        return [t for t in self.tracks.values() if t.is_confirmed]

    def reset(self):
        self.tracks = {}
        self.frame_count = 0
