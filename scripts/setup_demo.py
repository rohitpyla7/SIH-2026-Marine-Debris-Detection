"""
Seeds the local database with simulation survey frames and executes baseline pipeline validation.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def setup_demo():
    print("Setting up simulation survey environment...")

    from services.mission_service import (
        init_database, create_mission, save_image_record,
        save_detection, get_statistics
    )
    init_database()

    from simulation.synthetic_sonar import generate_demo_mission, DEMO_SCENARIOS
    Path("data/demo").mkdir(parents=True, exist_ok=True)
    mission_meta = generate_demo_mission(output_dir="data/demo")
    print(f"Generated {mission_meta['total_images']} simulation frames in data/demo/")

    create_mission(
        mission_id="MISSION-001",
        name="Survey Grid Alpha",
        date="2026-08-31",
        area="Coastal Survey Sector",
        operator="Autonomous Controller",
        data_label="SIMULATION",
    )

    from services.pipeline_service import get_pipeline
    pipeline = get_pipeline()
    total_detections = 0

    for img_meta in mission_meta["images"]:
        img_path = img_meta["filepath"]
        image_id = img_meta["image_id"]
        scenario_type = img_meta["scenario_type"]
        annotations = img_meta.get("annotations", [])
        lat = img_meta.get("lat")
        lon = img_meta.get("lon")
        depth_m = img_meta.get("depth_m")

        try:
            img = cv2.imread(img_path)
            if img is None:
                continue

            result = pipeline.run(
                img,
                image_id=image_id,
                scenario_type=scenario_type,
                annotations=annotations,
                lat=lat, lon=lon, depth_m=depth_m,
            )

            save_image_record(
                mission_id="MISSION-001",
                image_id=image_id,
                filename=img_meta["filename"],
                filepath=img_path,
                scenario_type=scenario_type,
                description=img_meta.get("description", ""),
                quality_label=result.quality.get("quality", "UNKNOWN"),
                quality_score=result.quality.get("score", 0.0),
                lat=lat, lon=lon, depth_m=depth_m,
                data_label="SIMULATION",
            )

            for det in result.detections:
                save_detection(
                    mission_id="MISSION-001",
                    image_id=image_id,
                    det=det,
                    lat=lat, lon=lon, depth_m=depth_m,
                )
                total_detections += 1

            result_dir = Path("results/predictions")
            result_dir.mkdir(parents=True, exist_ok=True)
            result_path = result_dir / f"result_{img_meta['filename']}"
            cv2.imwrite(str(result_path), result.annotated_image)

        except Exception as e:
            print(f"Error processing {img_meta['filename']}: {e}")

    stats = get_statistics()
    print("Simulation setup complete.")
    print(f"Missions: {stats['total_missions']} | Images: {stats['total_images']} | Detections: {stats['total_detections']}")


if __name__ == "__main__":
    setup_demo()
