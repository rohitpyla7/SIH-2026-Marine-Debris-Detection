from pathlib import Path
from PIL import Image
import random
import shutil

ROOT = Path("data/SubPipeMini2/SubPipeMiniSSS/DATA")
OUT = Path("data/subpipe_full_yolo")

TEST_DIR = Path("data/subpipe_hf_yolo/images/test")
TEST_STEMS = {p.stem for p in TEST_DIR.glob("*.png")}

random.seed(42)

# Collect HF + LF samples
samples = {}

for freq in ["HF", "LF"]:
    base = ROOT / f"SSS_{freq}_images"
    img_dir = base / "Image"
    lab_dir = base / "YOLO_Annotation"

    for img in img_dir.glob("*.pbm"):
        stem = img.stem

        # Timestamp-only files
        try:
            ts = float(stem)
        except ValueError:
            continue

        # Protect the existing test temporal window
        if stem in TEST_STEMS:
            continue
        if 1693569945.939 <= ts <= 1693570072.970:
            continue

        label = lab_dir / f"{stem}.txt"

        if label.exists() and label.read_text().strip():
            kind = "pos"
        elif not label.exists():
            kind = "neg"
        else:
            continue

        samples.setdefault(stem, []).append((freq, img, label, kind))

# Group by timestamp so HF/LF cannot cross train/val
groups = list(samples.items())
random.shuffle(groups)

n = len(groups)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

splits = {
    "train": groups[:train_end],
    "val": groups[train_end:val_end],
}

# Existing test is copied from the already prepared dataset.
for split in ["train", "val", "test"]:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

# Copy existing test unchanged
for img in (Path("data/subpipe_hf_yolo/images/test")).glob("*.png"):
    shutil.copy2(img, OUT / "images" / "test" / img.name)

for lab in (Path("data/subpipe_hf_yolo/labels/test")).glob("*.txt"):
    shutil.copy2(lab, OUT / "labels" / "test" / lab.name)

stats = {}

for split, split_groups in splits.items():
    pos = neg = boxes = 0

    for stem, entries in split_groups:
        for freq, img, label, kind in entries:
            out_stem = f"{freq}_{stem}"
            out_img = OUT / "images" / split / f"{out_stem}.png"
            out_lab = OUT / "labels" / split / f"{out_stem}.txt"

            Image.open(img).convert("L").save(out_img)

            if kind == "pos":
                text = label.read_text().strip()
                out_lab.write_text(text + "\n")
                pos += 1
                boxes += len(text.splitlines())
            else:
                out_lab.write_text("")
                neg += 1

    stats[split] = (pos, neg, boxes)

yaml = OUT / "data.yaml"
yaml.write_text(
    f"""path: {OUT.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: Pipeline
"""
)

print("\nDATASET CREATED")
print("Groups:", n)
for split, (pos, neg, boxes) in stats.items():
    print(f"{split}: {pos} positive, {neg} negative, {boxes} boxes")

print("test: 127 existing images")
print("Location:", OUT.resolve())
