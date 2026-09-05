"""
Model evaluation script for calculating validation and test metrics (mAP, Precision, Recall).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Sonar Target Detector Evaluation")
    p.add_argument("--model", default="models/best.pt", help="Path to trained weights")
    p.add_argument("--data", default=None, help="Path to data.yaml")
    p.add_argument("--split", default="test", help="Dataset split ('test', 'val')")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--output", default="reports/evaluation_results.json", help="Destination JSON path")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Run training first via: python train.py")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics is required. Run: pip install ultralytics")
        sys.exit(1)

    data_yaml = args.data
    if not data_yaml:
        from datasets.adapter import DatasetAdapter
        adapter = DatasetAdapter()
        data_yaml = adapter.get_data_yaml_path()
        if not data_yaml:
            print("Error: No data.yaml path specified or discovered. Use --data <path>")
            sys.exit(1)

    print(f"Evaluating model: {model_path} against {data_yaml} ({args.split} split)")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=data_yaml,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        verbose=True,
        plots=True,
    )

    results = {
        "model": str(model_path),
        "dataset": data_yaml,
        "split": args.split,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
            "f1": round(float(metrics.box.f1.mean()), 4) if hasattr(metrics.box, "f1") else None,
            "map50": round(float(metrics.box.map50), 4),
            "map50_95": round(float(metrics.box.map), 4),
            "classes": metrics.names if hasattr(metrics, "names") else {},
        },
        "config": {
            "conf_threshold": args.conf,
            "iou_threshold": args.iou,
            "imgsz": args.imgsz,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation metrics saved to: {out_path}")


if __name__ == "__main__":
    main()
