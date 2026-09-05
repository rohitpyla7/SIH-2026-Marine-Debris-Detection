"""
Headless CLI inference script for processing individual sonar images or batch collections.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import cv2

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Sonar Target Inference CLI")
    p.add_argument("--image", required=True, help="Path to input sonar image")
    p.add_argument("--model", default="models/best.pt", help="Path to model weights")
    p.add_argument("--demo", action="store_true", help="Force fallback demo mode")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--out", default="outputs/detections", help="Output directory")
    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--depth", type=float, default=None)
    p.add_argument("--mission", default="MISSION-CLI")
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Unable to load image: {args.image}")
        sys.exit(1)

    h, w = img.shape[:2]
    stem = Path(args.image).stem
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from utils.quality import assess_quality
    quality = assess_quality(img)

    from utils.preprocessing import preprocess_image
    prep_result = preprocess_image(img)
    preprocessed = prep_result["preprocessed"]
    cv2.imwrite(str(out_dir / f"{stem}_preprocessed.png"), preprocessed)

    from services.pipeline_service import SonarAnalysisPipeline
    pipeline = SonarAnalysisPipeline(model_path=args.model if not args.demo else None)

    result = pipeline.run(
        img,
        image_id=f"{stem}_{int(time.time())}",
        lat=args.lat,
        lon=args.lon,
        depth_m=args.depth,
    )

    ann_path = out_dir / f"{stem}_detected.jpg"
    cv2.imwrite(str(ann_path), result.annotated_image)

    output = {
        "model": pipeline.detector.mode,
        "mode": pipeline.detector.mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image": {
            "path": args.image,
            "width": w,
            "height": h,
            "preprocessed": str(out_dir / f"{stem}_preprocessed.png"),
            "detected": str(ann_path),
        },
        "quality": result.quality,
        "detections": [d.to_dict() for d in result.detections],
        "anomaly": result.anomaly_result,
        "timing": result.timing,
        "warnings": result.warnings,
    }

    json_path = out_dir / f"{stem}_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    total_ms = round((time.time() - t_start) * 1000)
    print(f"Completed inference in {total_ms} ms | Detections: {len(result.detections)}")
    print(f"Saved: {ann_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
