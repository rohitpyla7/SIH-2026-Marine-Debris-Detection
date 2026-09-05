"""
YOLOv8 training pipeline for side-scan sonar target detection.
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="Side-Scan Sonar YOLOv8 Training")
    p.add_argument("--data", default=None, help="Path to data.yaml dataset config")
    p.add_argument("--epochs", type=int, default=50, help="Epoch count")
    p.add_argument("--imgsz", type=int, default=640, help="Input resolution")
    p.add_argument("--batch", type=int, default=8, help="Batch size")
    p.add_argument("--device", default="auto", help="Device target ('cpu', '0', 'auto')")
    p.add_argument("--project", default="runs/train", help="Run directory")
    p.add_argument("--name", default="sonar_yolov8n", help="Experiment name")
    p.add_argument("--resume", action="store_true", help="Resume checkpoint")
    p.add_argument("--inspect-only", action="store_true", help="Inspect dataset without starting training")
    return p.parse_args()


def check_ultralytics() -> bool:
    try:
        from ultralytics import YOLO
        return True
    except ImportError:
        print("Error: ultralytics is required. Run: pip install ultralytics")
        return False


def resolve_data_yaml(args) -> str:
    if args.data and Path(args.data).exists():
        return args.data

    from datasets.adapter import DatasetAdapter
    adapter = DatasetAdapter()
    yaml_path = adapter.get_data_yaml_path()
    if yaml_path:
        return yaml_path

    print("[Info] No custom data.yaml provided. Setting up synthetic benchmark config.")
    return create_demo_data_yaml()


def create_demo_data_yaml() -> str:
    import json
    import cv2
    import yaml
    from simulation.synthetic_sonar import generate_demo_mission

    demo_dir = Path("data/demo")
    if not (demo_dir / "MISSION-001_metadata.json").exists():
        print("[Info] Generating simulation benchmark dataset...")
        generate_demo_mission(output_dir="data/demo")

    labels_dir = demo_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    with open(demo_dir / "MISSION-001_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)

    classes = [
        "fishing_net", "metal_debris", "plastic_debris",
        "large_structure", "unknown_anomaly", "tracking_object"
    ]
    class_to_id = {c: i for i, c in enumerate(classes)}

    for img_meta in meta["images"]:
        img_path = Path(img_meta["filepath"])
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_file = labels_dir / (img_path.stem + ".txt")
        lines = []
        for ann in img_meta.get("annotations", []):
            cn = ann.get("class_name", "")
            cid = class_to_id.get(cn, 0)
            bbox = ann.get("bbox", [0, 0, 10, 10])
            x1, y1, x2, y2 = bbox
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        label_file.write_text("\n".join(lines), encoding="utf-8")

    data_yaml = {
        "path": str(demo_dir.absolute()),
        "train": ".",
        "val": ".",
        "nc": len(classes),
        "names": classes,
    }
    yaml_path = demo_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f)

    return str(yaml_path)


def inspect_dataset(data_yaml_path: str):
    import yaml
    from datasets.adapter import DatasetAdapter

    try:
        with open(data_yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        dataset_root = data.get("path", ".")
        adapter = DatasetAdapter(dataset_path=dataset_root, mode="custom")
        info = adapter.inspect()
        print(f"Dataset Path: {info.get('dataset_path')}")
        print(f"Total Frames: {info.get('total_images', 'N/A')}")
        print(f"Annotations:  {info.get('total_annotations', 'N/A')}")
        print("Classes:")
        for cls, count in (info.get("classes") or {}).items():
            print(f"  - {cls}: {count}")
    except Exception as e:
        print(f"Inspection error: {e}")


def main():
    args = parse_args()
    print("Starting YOLO training pipeline")
    print(f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    if not check_ultralytics():
        sys.exit(1)

    from ultralytics import YOLO

    data_yaml = resolve_data_yaml(args)
    print(f"Using Dataset YAML: {data_yaml}")
    inspect_dataset(data_yaml)

    if args.inspect_only:
        return

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=20,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if best_pt.exists():
        dest = Path("models/best.pt")
        dest.parent.mkdir(exist_ok=True)
        import shutil
        shutil.copy(best_pt, dest)
        print(f"Trained model saved to: {dest}")


if __name__ == "__main__":
    main()
