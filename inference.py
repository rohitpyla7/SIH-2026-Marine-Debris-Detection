import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from services.pipeline_service import SonarAnalysisPipeline
from utils.preprocessing import preprocess_image
from utils.quality import assess_quality


def main():
    parser = argparse.ArgumentParser(
        description="Run underwater sonar analysis pipeline"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input sonar image"
    )

    parser.add_argument(
        "--model",
        default="models/best.pt",
        help="Path to YOLO model"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo/rule-based mode"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold"
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="YOLO IoU threshold"
    )

    parser.add_argument(
        "--output",
        default="outputs/detections",
        help="Output directory"
    )

    parser.add_argument(
        "--lat",
        type=float,
        default=None
    )

    parser.add_argument(
        "--lon",
        type=float,
        default=None
    )

    parser.add_argument(
        "--depth",
        type=float,
        default=None
    )

    parser.add_argument(
        "--mission",
        default=None
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # LOAD IMAGE
    # ---------------------------------------------------------

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    print(f"Input image: {image_path}")

    # ---------------------------------------------------------
    # QUALITY
    # ---------------------------------------------------------

    quality = assess_quality(img)

    print(f"Quality score: {quality}")

    # ---------------------------------------------------------
    # PREPROCESSING PREVIEW
    # ---------------------------------------------------------

    prep_result = preprocess_image(img)

    output_dir = Path(args.output)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    preprocessed_path = (
        output_dir /
        f"{image_path.stem}_preprocessed.png"
    )

    if isinstance(prep_result, dict):
        preprocessed_img = prep_result.get(
            "preprocessed",
            img
        )
    elif isinstance(prep_result, tuple):
        preprocessed_img = prep_result[0]
    else:
        preprocessed_img = prep_result

    if isinstance(preprocessed_img, np.ndarray):
        cv2.imwrite(
            str(preprocessed_path),
            preprocessed_img
        )
        print(
            f"Saved preprocessed image: "
            f"{preprocessed_path}"
        )

    # ---------------------------------------------------------
    # PIPELINE
    # ---------------------------------------------------------

    pipeline = SonarAnalysisPipeline(
        model_path=(
            args.model
            if not args.demo
            else None
        ),
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    # ---------------------------------------------------------
    # RUN ANALYSIS
    # ---------------------------------------------------------

    result = pipeline.run(
        img,
        image_id=image_path.stem,
        lat=args.lat,
        lon=args.lon,
        depth_m=args.depth,
    )

    # ---------------------------------------------------------
    # SAVE JSON
    # ---------------------------------------------------------

    result_path = (
        output_dir /
        f"{image_path.stem}_result.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            result.to_dict(),
            f,
            indent=2,
            default=str
        )

    print(
        f"Saved: {result_path}"
    )

    # ---------------------------------------------------------
    # SAVE ANNOTATED IMAGE
    # ---------------------------------------------------------

    annotated_path = (
        output_dir /
        f"{image_path.stem}_annotated.png"
    )

    cv2.imwrite(
        str(annotated_path),
        result.annotated_image
    )

    print(
        f"Saved annotated image: "
        f"{annotated_path}"
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\n=== ANALYSIS RESULT ===")

    print(
        f"Mode: {result.mode}"
    )

    print(
        f"Model: {result.model_version}"
    )

    print(
        f"Detections: "
        f"{len(result.detections)}"
    )

    print(
        f"Known objects: "
        f"{result.num_known}"
    )

    print(
        f"Anomalies: "
        f"{result.num_anomalies}"
    )

    if result.detections:
        for i, detection in enumerate(
            result.detections,
            start=1
        ):
            print(
                f"\nDetection {i}:"
            )

            print(
                f"  Class: "
                f"{detection.display_name}"
            )

            print(
                f"  Confidence: "
                f"{detection.confidence:.4f}"
            )

            print(
                f"  Evidence: "
                f"{detection.evidence_pct:.1f}%"
            )

            print(
                f"  Severity: "
                f"{detection.severity}"
            )

            print(
                f"  Shadow score: "
                f"{detection.shadow_score:.4f}"
            )

            print(
                f"  Anomaly score: "
                f"{detection.anomaly_score:.4f}"
            )

    if result.warnings:
        print("\nWarnings:")

        for warning in result.warnings:
            print(
                f"  - {warning}"
            )


if __name__ == "__main__":
    main()