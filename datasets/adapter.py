"""
Dataset adapter interface for YOLO-format side-scan sonar datasets and synthetic test data.
Inspects splits, class labels, and annotation distribution directly from dataset sources.
"""

import csv
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml

DEMO_DATASET_PATH = Path("data/demo")
SUBPIPE_PATH_ENV = os.environ.get("SUBPIPE_PATH", "")


class DatasetAdapter:
    """Dataset adapter supporting standard YOLO structures and fallback synthetic benchmarks."""

    def __init__(self, dataset_path: Optional[str] = None, mode: str = "auto"):
        self.mode = mode
        self.dataset_path = None
        self.classes = []
        self.class_ids = {}
        self._detected_format = None

        if mode == "demo" or (mode == "auto" and not dataset_path and not SUBPIPE_PATH_ENV):
            self.mode = "demo"
            self.dataset_path = DEMO_DATASET_PATH
        else:
            raw = dataset_path or SUBPIPE_PATH_ENV
            if raw:
                self.dataset_path = Path(raw)
                self.mode = "subpipe" if mode == "auto" else mode
            else:
                self.mode = "demo"
                self.dataset_path = DEMO_DATASET_PATH

    def inspect(self) -> dict:
        """Inspects and counts class distribution and split counts."""
        if self.mode == "demo":
            return self._inspect_demo()
        return self._inspect_yolo_dataset()

    def _inspect_demo(self) -> dict:
        meta_file = DEMO_DATASET_PATH / "MISSION-001_metadata.json"
        if not meta_file.exists():
            return {
                "mode": "demo",
                "status": "NOT_FOUND",
                "note": "Run python scripts/setup_demo.py to generate demo data",
            }
        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)
        images = meta.get("images", [])
        all_classes = {}
        for img in images:
            for ann in img.get("annotations", []):
                cn = ann.get("class_name", "unknown")
                all_classes[cn] = all_classes.get(cn, 0) + 1
        return {
            "mode": "demo",
            "status": "OK",
            "dataset_path": str(DEMO_DATASET_PATH),
            "total_images": len(images),
            "total_annotations": sum(all_classes.values()),
            "classes": all_classes,
            "note": "Simulation benchmark records",
            "data_label": "SIMULATION",
        }

    def _inspect_yolo_dataset(self) -> dict:
        root = self.dataset_path
        if not root or not root.exists():
            return {
                "mode": self.mode,
                "status": "NOT_FOUND",
                "dataset_path": str(root),
                "note": f"Dataset directory not found: {root}",
            }

        result = {
            "mode": self.mode,
            "status": "OK",
            "dataset_path": str(root),
            "classes": {},
            "class_ids": {},
            "splits": {},
            "total_images": 0,
            "total_annotations": 0,
            "data_label": "REAL_SSS",
        }

        yaml_paths = list(root.rglob("data.yaml")) + list(root.rglob("dataset.yaml"))
        if yaml_paths:
            with open(yaml_paths[0], encoding="utf-8") as f:
                data_yaml = yaml.safe_load(f)
            names = data_yaml.get("names", [])
            if isinstance(names, list):
                result["class_ids"] = {i: n for i, n in enumerate(names)}
                result["classes_from_yaml"] = names
                result["nc"] = len(names)
            elif isinstance(names, dict):
                result["class_ids"] = {int(k): v for k, v in names.items()}
                result["classes_from_yaml"] = list(names.values())
                result["nc"] = len(names)
            result["yaml_path"] = str(yaml_paths[0])
        else:
            result["yaml_warning"] = "data.yaml not found"

        class_counter = {}
        total_images = 0
        total_annotations = 0

        for split in ["train", "val", "test", "valid"]:
            label_dirs = [
                root / split / "labels",
                root / "labels" / split,
            ]
            for label_dir in label_dirs:
                if not label_dir.exists():
                    continue
                txt_files = list(label_dir.glob("*.txt"))
                split_count = 0
                split_annotations = 0
                for txt in txt_files:
                    total_images += 1
                    split_count += 1
                    try:
                        lines = txt.read_text(encoding="utf-8").strip().split("\n")
                        for line in lines:
                            if not line.strip():
                                continue
                            parts = line.split()
                            if len(parts) >= 5:
                                cls_id = int(parts[0])
                                cls_name = result["class_ids"].get(cls_id, f"class_{cls_id}")
                                class_counter[cls_name] = class_counter.get(cls_name, 0) + 1
                                total_annotations += 1
                                split_annotations += 1
                    except Exception:
                        pass
                if split_count > 0:
                    result["splits"][split] = {
                        "images": split_count,
                        "annotations": split_annotations,
                        "label_dir": str(label_dir),
                    }
                break

        result["classes"] = class_counter
        result["total_images"] = total_images
        result["total_annotations"] = total_annotations

        img_dims = []
        for img_dir in [root / "images", root]:
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                for img_path in list(img_dir.rglob(ext))[:5]:
                    try:
                        import cv2
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img_dims.append(img.shape[:2])
                    except Exception:
                        pass
            if img_dims:
                break
        if img_dims:
            result["sample_image_dims"] = img_dims

        return result

    def get_data_yaml_path(self) -> Optional[str]:
        if self.mode == "demo":
            return None
        root = self.dataset_path
        yaml_paths = list(root.rglob("data.yaml")) if root and root.exists() else []
        return str(yaml_paths[0]) if yaml_paths else None

    def is_available(self) -> bool:
        if self.mode == "demo":
            return (DEMO_DATASET_PATH / "MISSION-001_metadata.json").exists()
        return self.dataset_path is not None and self.dataset_path.exists()

    def summary_str(self) -> str:
        info = self.inspect()
        if info["status"] != "OK":
            return f"Dataset unavailable: {info.get('note', '')}"
        classes = info.get("classes", {})
        total = info.get("total_images", 0)
        return f"Mode={self.mode} | Frames={total} | Classes={list(classes.keys())}"
