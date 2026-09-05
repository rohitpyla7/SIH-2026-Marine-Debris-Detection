"""
Synthetic acoustic side-scan sonar image generator for testing and simulation.
Simulates speckle backscatter noise, central nadir track, bright returns, and acoustic shadows.
"""

import json
import math
import os
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np

SYNTHETIC_LABEL = "SYNTHETIC_SIMULATION"


def make_seabed_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Generate speckled seabed backscatter texture with central nadir zone."""
    base = np.ones((h, w), dtype=np.float32) * 80.0
    center = w // 2
    for x in range(w):
        dist = abs(x - center) / (w / 2)
        base[:, x] *= (0.4 + 0.6 * dist)

    # Multiplicative speckle modeling
    speckle = rng.gamma(shape=2.0, scale=0.5, size=(h, w)).astype(np.float32)
    base = base * speckle

    low_freq = cv2.GaussianBlur(
        rng.normal(0, 15, (h, w)).astype(np.float32), (51, 51), 0
    )
    base = base + low_freq

    # Nadir specular return
    nadir_w = max(3, w // 60)
    base[:, center - nadir_w : center + nadir_w] = 220.0

    return np.clip(base, 0, 255).astype(np.uint8)


def add_object_with_shadow(
    img: np.ndarray,
    cx: int,
    cy: int,
    obj_w: int,
    obj_h: int,
    shape: str = "rect",
    intensity: float = 220.0,
    shadow_length: int = 60,
    angle_deg: float = 0.0,
    label: Optional[str] = None,
) -> dict:
    """Renders acoustic reflection and cast shadow for a synthetic target."""
    h, w = img.shape[:2]
    canvas = img.copy() if img.ndim == 2 else img

    obj_mask = np.zeros((h, w), dtype=np.uint8)
    if shape == "rect":
        x1 = max(0, cx - obj_w // 2)
        y1 = max(0, cy - obj_h // 2)
        x2 = min(w - 1, cx + obj_w // 2)
        y2 = min(h - 1, cy + obj_h // 2)
        cv2.rectangle(obj_mask, (x1, y1), (x2, y2), 255, -1)
    elif shape == "ellipse":
        cv2.ellipse(obj_mask, (cx, cy), (obj_w // 2, obj_h // 2),
                    angle_deg, 0, 360, 255, -1)
    elif shape == "net":
        pts = []
        n_pts = 8
        for i in range(n_pts):
            a = 2 * math.pi * i / n_pts + math.radians(angle_deg)
            r = (obj_w // 2) * (0.7 + 0.3 * random.random())
            pts.append([int(cx + r * math.cos(a)), int(cy + r * math.sin(a))])
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(obj_mask, [pts], 255)
    elif shape == "pipe":
        pts = cv2.boxPoints(((cx, cy), (obj_w, obj_h // 3), angle_deg))
        pts = np.intp(pts)
        cv2.fillPoly(obj_mask, [pts], 255)

    bright_region = obj_mask > 0
    canvas[bright_region] = np.clip(
        canvas[bright_region].astype(np.float32) * 0.3 + intensity * 0.7, 0, 255
    ).astype(np.uint8)

    nadir_x = w // 2
    shadow_dir = 1 if cx > nadir_x else -1

    shadow_mask = np.zeros((h, w), dtype=np.uint8)
    sh_x1 = cx + shadow_dir * (obj_w // 2)
    sh_x2 = cx + shadow_dir * (obj_w // 2 + shadow_length)
    sh_y1 = cy - obj_h // 2
    sh_y2 = cy + obj_h // 2

    pts_shadow = np.array([
        [sh_x1, sh_y1], [sh_x2, sh_y1 - shadow_length // 4],
        [sh_x2, sh_y2 + shadow_length // 4], [sh_x1, sh_y2]
    ], dtype=np.int32)

    cv2.fillPoly(shadow_mask, [pts_shadow], 255)
    shadow_region = shadow_mask > 0
    canvas[shadow_region] = np.clip(
        canvas[shadow_region].astype(np.float32) * 0.25, 0, 50
    ).astype(np.uint8)

    ys, xs = np.where(obj_mask > 0)
    if len(xs) == 0:
        bbox = [cx - obj_w // 2, cy - obj_h // 2, cx + obj_w // 2, cy + obj_h // 2]
    else:
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    sys_, sxs = np.where(shadow_mask > 0)
    shadow_bbox = [int(sxs.min()), int(sys_.min()), int(sxs.max()), int(sys_.max())] if len(sxs) > 0 else None

    return {
        "label": label,
        "bbox": bbox,
        "shadow_bbox": shadow_bbox,
        "center": [cx, cy],
        "shape": shape,
    }


def generate_scenario(
    scenario_id: int,
    scenario_type: str,
    h: int = 512,
    w: int = 1024,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """Generates a synthetic sonar frame corresponding to defined scenario type."""
    if seed is None:
        seed = scenario_id * 42 + 7
    rng = np.random.default_rng(seed)
    random.seed(seed)

    img = make_seabed_background(h, w, rng)
    annotations = []

    if scenario_type == "normal_seabed":
        description = "Nominal seabed backscatter without obstruction"
        quality = "GOOD"

    elif scenario_type == "fishing_net":
        info = add_object_with_shadow(
            img, cx=600, cy=256, obj_w=160, obj_h=80, shape="net",
            intensity=215.0, shadow_length=80, label="fishing_net"
        )
        annotations.append(info)
        description = "Entangled fishing net with distinct trailing acoustic shadow"
        quality = "GOOD"

    elif scenario_type == "metal_debris":
        info = add_object_with_shadow(
            img, cx=650, cy=200, obj_w=60, obj_h=60, shape="ellipse",
            intensity=235.0, shadow_length=70, label="metal_debris"
        )
        annotations.append(info)
        info2 = add_object_with_shadow(
            img, cx=750, cy=320, obj_w=40, obj_h=40, shape="rect",
            intensity=225.0, shadow_length=50, label="metal_debris"
        )
        annotations.append(info2)
        description = "High-reflectivity metallic debris with sharp acoustic contrast"
        quality = "GOOD"

    elif scenario_type == "plastic_debris":
        info = add_object_with_shadow(
            img, cx=580, cy=280, obj_w=90, obj_h=40, shape="rect",
            intensity=195.0, shadow_length=45, label="plastic_debris"
        )
        annotations.append(info)
        description = "Submerged polymeric debris"
        quality = "GOOD"

    elif scenario_type == "large_structure":
        info = add_object_with_shadow(
            img, cx=620, cy=256, obj_w=220, obj_h=120, shape="rect",
            intensity=240.0, shadow_length=150, label="shipwreck"
        )
        annotations.append(info)
        description = "Large seabed structure / wreck feature"
        quality = "GOOD"

    elif scenario_type == "unknown_anomaly":
        info = add_object_with_shadow(
            img, cx=680, cy=256, obj_w=55, obj_h=110, shape="ellipse",
            intensity=200.0, shadow_length=60, angle_deg=45.0, label="unknown_anomaly"
        )
        annotations.append(info)
        description = "Unclassified acoustic signature with anomalous aspect ratio"
        quality = "GOOD"

    elif scenario_type == "low_quality":
        noise = rng.normal(0, 45, (h, w)).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        info = add_object_with_shadow(
            img, cx=600, cy=256, obj_w=70, obj_h=50, shape="rect",
            intensity=180.0, shadow_length=50, label="metal_debris"
        )
        annotations.append(info)
        description = "Degraded acoustic frame under high turbidity and noise"
        quality = "POOR"

    elif scenario_type == "multi_object":
        info1 = add_object_with_shadow(
            img, cx=500, cy=180, obj_w=80, obj_h=50, shape="net",
            intensity=210.0, shadow_length=60, label="fishing_net"
        )
        annotations.append(info1)
        info2 = add_object_with_shadow(
            img, cx=700, cy=300, obj_w=50, obj_h=50, shape="ellipse",
            intensity=230.0, shadow_length=55, label="metal_debris"
        )
        annotations.append(info2)
        info3 = add_object_with_shadow(
            img, cx=820, cy=200, obj_w=40, obj_h=30, shape="rect",
            intensity=190.0, shadow_length=30, label="plastic_debris"
        )
        annotations.append(info3)
        description = "Multiple debris targets in survey track"
        quality = "GOOD"

    elif scenario_type == "strong_shadow":
        info = add_object_with_shadow(
            img, cx=600, cy=256, obj_w=80, obj_h=80, shape="rect",
            intensity=245.0, shadow_length=140, label="metal_debris"
        )
        annotations.append(info)
        description = "Target casting extended acoustic shadow"
        quality = "GOOD"

    elif scenario_type == "tracking_sequence":
        offset = (scenario_id % 3) * 15
        info = add_object_with_shadow(
            img, cx=600 + offset, cy=256, obj_w=70, obj_h=50, shape="net",
            intensity=210.0, shadow_length=65, label="fishing_net"
        )
        annotations.append(info)
        description = f"Tracking sequence frame {(scenario_id % 3) + 1}/3"
        quality = "GOOD"

    else:
        description = "Unspecified scenario"
        quality = "UNKNOWN"

    metadata = {
        "scenario_id": scenario_id,
        "scenario_type": scenario_type,
        "description": description,
        "image_quality": quality,
        "data_label": SYNTHETIC_LABEL,
        "annotations": annotations,
        "image_size": [h, w],
        "seed": seed,
    }

    return img, metadata


DEMO_SCENARIOS = [
    (1,  "normal_seabed",     "IMG-001_normal_seabed.png"),
    (2,  "fishing_net",       "IMG-002_fishing_net.png"),
    (3,  "metal_debris",      "IMG-003_metal_debris.png"),
    (4,  "plastic_debris",    "IMG-004_plastic_debris.png"),
    (5,  "large_structure",   "IMG-005_large_structure.png"),
    (6,  "unknown_anomaly",   "IMG-006_unknown_anomaly.png"),
    (7,  "low_quality",       "IMG-007_low_quality.png"),
    (8,  "multi_object",      "IMG-008_multi_object.png"),
    (9,  "strong_shadow",     "IMG-009_strong_shadow.png"),
    (10, "tracking_sequence", "IMG-010_tracking_a.png"),
    (11, "tracking_sequence", "IMG-011_tracking_b.png"),
    (12, "tracking_sequence", "IMG-012_tracking_c.png"),
]

DEMO_COORDS = [
    {"lat": 10.8505, "lon": 76.2711},
    {"lat": 10.8510, "lon": 76.2720},
    {"lat": 10.8515, "lon": 76.2730},
    {"lat": 10.8520, "lon": 76.2740},
    {"lat": 10.8525, "lon": 76.2750},
    {"lat": 10.8530, "lon": 76.2760},
    {"lat": 10.8535, "lon": 76.2770},
    {"lat": 10.8540, "lon": 76.2780},
    {"lat": 10.8545, "lon": 76.2790},
    {"lat": 10.8550, "lon": 76.2800},
    {"lat": 10.8555, "lon": 76.2810},
    {"lat": 10.8560, "lon": 76.2820},
]


def generate_demo_mission(output_dir: str = "datasets/raw") -> dict:
    """Creates synthetic mission dataset and corresponding metadata catalogue."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mission_images = []

    for idx, (sid, stype, fname) in enumerate(DEMO_SCENARIOS):
        img, meta = generate_scenario(sid, stype)
        img_path = output_path / fname
        cv2.imwrite(str(img_path), img)

        coords = DEMO_COORDS[idx] if idx < len(DEMO_COORDS) else DEMO_COORDS[-1]

        mission_images.append({
            "image_id": f"IMG-{idx + 1:03d}",
            "filename": fname,
            "filepath": str(img_path),
            "scenario_type": stype,
            "description": meta["description"],
            "image_quality": meta["image_quality"],
            "annotations": meta["annotations"],
            "lat": coords["lat"],
            "lon": coords["lon"],
            "depth_m": round(15.0 + idx * 0.5, 1),
            "data_label": SYNTHETIC_LABEL,
            "coordinates_label": "Simulated Coordinates",
        })

    mission = {
        "mission_id": "MISSION-001",
        "name": "Coastal Sonar Survey Simulation",
        "date": "2026-08-31",
        "area": "Survey Grid Alpha",
        "survey_lat_center": 10.853,
        "survey_lon_center": 76.275,
        "vehicle": "AUV-Surveyor-01",
        "sonar_config": "Side-Scan Sonar 450kHz",
        "operator": "Automated Survey Controller",
        "status": "completed",
        "data_label": SYNTHETIC_LABEL,
        "coordinates_label": "Simulated Coordinates",
        "images": mission_images,
        "total_images": len(mission_images),
    }

    meta_path = output_path / "MISSION-001_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(mission, f, indent=2)

    return mission


if __name__ == "__main__":
    mission = generate_demo_mission()
    print(f"Generated {mission['total_images']} simulation frames.")
