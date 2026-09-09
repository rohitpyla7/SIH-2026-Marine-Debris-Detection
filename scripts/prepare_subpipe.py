from pathlib import Path
import shutil
import cv2
import yaml

# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT
    / "data"
    / "SubPipeMini2"
    / "SubPipeMiniSSS"
    / "DATA"
    / "SSS_HF_images"
)

OUTPUT = ROOT / "data" / "subpipe_hf_yolo"

IMAGE_SOURCE = SOURCE / "Image"
LABEL_SOURCE = SOURCE / "YOLO_Annotation"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

# Main continuous temporal chunk
images = sorted(
    IMAGE_SOURCE.glob("*.pbm"),
    key=lambda p: float(p.stem)
)

# First 842 images = main continuous chunk
images = images[:842]

# Temporal split: 70 / 15 / 15
n = len(images)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

splits = {
    "train": images[:train_end],
    "val": images[train_end:val_end],
    "test": images[val_end:n],
}


# --------------------------------------------------
# Prepare output directories
# --------------------------------------------------

for split in splits:
    (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Process images
# --------------------------------------------------

summary = {}

for split, split_images in splits.items():

    positive = 0
    negative = 0
    boxes = 0

    for pbm_path in split_images:

        # Read sonar PBM
        img = cv2.imread(str(pbm_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise RuntimeError(f"Could not read image: {pbm_path}")

        # Save as standard PNG
        output_image = OUTPUT / "images" / split / f"{pbm_path.stem}.png"
        cv2.imwrite(str(output_image), img)

        # Matching YOLO annotation
        label_path = LABEL_SOURCE / f"{pbm_path.stem}.txt"
        output_label = OUTPUT / "labels" / split / f"{pbm_path.stem}.txt"

        if label_path.exists():

            text = label_path.read_text(encoding="utf-8").strip()

            if text:
                output_label.write_text(text + "\n", encoding="utf-8")

                positive += 1
                boxes += len(text.splitlines())

            else:
                output_label.write_text("", encoding="utf-8")
                negative += 1

        else:
            # Verified negative/background image
            output_label.write_text("", encoding="utf-8")
            negative += 1

    summary[split] = {
        "images": len(split_images),
        "positive": positive,
        "negative": negative,
        "boxes": boxes,
    }


# --------------------------------------------------
# Create YOLO data.yaml
# --------------------------------------------------

data_yaml = {
    "path": str(OUTPUT.resolve()),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 1,
    "names": ["Pipeline"],
}

with open(OUTPUT / "data.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(data_yaml, f, sort_keys=False)


# --------------------------------------------------
# Final report
# --------------------------------------------------

print()
print("=" * 55)
print("SUBPIPE HF DATASET PREPARATION COMPLETE")
print("=" * 55)

for split, info in summary.items():
    print(
        f"{split.upper():5} | "
        f"images={info['images']:3} | "
        f"positive={info['positive']:3} | "
        f"negative={info['negative']:3} | "
        f"boxes={info['boxes']:3}"
    )

print("-" * 55)
print(f"Output: {OUTPUT}")
print(f"YAML:   {OUTPUT / 'data.yaml'}")
print("=" * 55)