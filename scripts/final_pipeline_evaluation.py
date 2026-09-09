from pathlib import Path
import sys
import time
import json
import cv2
import numpy as np

from services.pipeline_service import SonarAnalysisPipeline


TEST_DIR = Path("data/subpipe_hf_yolo/images/test")
OUTPUT_DIR = Path("reports/final_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    images = sorted(
        list(TEST_DIR.glob("*.png")) +
        list(TEST_DIR.glob("*.jpg")) +
        list(TEST_DIR.glob("*.jpeg"))
    )

    if not images:
        raise RuntimeError(f"No test images found in {TEST_DIR}")

    print(f"Test images: {len(images)}")

    pipeline = SonarAnalysisPipeline(
        conf_threshold=0.02,
        iou_threshold=0.45,
    )

    results = []

    total_ms = 0.0
    total_detections = 0
    anomalous_images = 0
    shadow_scores = []

    for i, image_path in enumerate(images, 1):
        img = cv2.imread(str(image_path))

        if img is None:
            print(f"[SKIP] Could not read {image_path}")
            continue

        t0 = time.perf_counter()

        try:
            result = pipeline.run(
                img,
                image_id=image_path.stem,
            )
        except Exception as exc:
            print(f"[ERROR] {image_path.name}: {exc}")
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_ms += elapsed_ms

        detections = getattr(result, "detections", []) or []

        total_detections += len(detections)

        image_anomalous = False

        for det in detections:
            shadow = getattr(det, "shadow_score", None)

            if shadow is not None:
                shadow_scores.append(float(shadow))

            if getattr(det, "is_anomaly", False):
                image_anomalous = True

        if image_anomalous:
            anomalous_images += 1

        results.append({
            "image": image_path.name,
            "detections": len(detections),
            "anomalous": image_anomalous,
            "elapsed_ms": round(elapsed_ms, 2),
        })

        if i % 10 == 0 or i == len(images):
            print(f"[{i}/{len(images)}] processed")

    processed = len(results)

    summary = {
        "test_images": len(images),
        "processed_images": processed,
        "total_detections": total_detections,
        "images_with_anomalies": anomalous_images,
        "mean_pipeline_ms": round(
            total_ms / max(processed, 1),
            2,
        ),
        "mean_shadow_score": round(
            float(np.mean(shadow_scores)),
            4,
        ) if shadow_scores else None,
        "min_shadow_score": round(
            float(np.min(shadow_scores)),
            4,
        ) if shadow_scores else None,
        "max_shadow_score": round(
            float(np.max(shadow_scores)),
            4,
        ) if shadow_scores else None,
    }

    output = {
        "summary": summary,
        "images": results,
    }

    output_path = OUTPUT_DIR / "final_pipeline_evaluation.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print()
    print("=== FINAL PIPELINE EVALUATION ===")
    print(json.dumps(summary, indent=2))
    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

