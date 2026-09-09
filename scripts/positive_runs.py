import glob
import os

files = sorted(
    glob.glob("./data/subpipe_hf_yolo/labels/train/*.txt")
    + glob.glob("./data/subpipe_hf_yolo/labels/val/*.txt")
    + glob.glob("./data/subpipe_hf_yolo/labels/test/*.txt"),
    key=lambda x: float(os.path.basename(x)[:-4])
)

positive = []

for f in files:
    with open(f) as file:
        if any(line.strip() for line in file):
            positive.append(float(os.path.basename(f)[:-4]))

runs = []

start = prev = positive[0]

for t in positive[1:]:
    if t - prev <= 2.0:
        prev = t
    else:
        runs.append((start, prev))
        start = prev = t

runs.append((start, prev))

print("POSITIVE RUNS:", len(runs))
print("-" * 60)

for i, (a, b) in enumerate(runs, 1):
    print(f"{i:02d}: {a:.3f} -> {b:.3f}   duration={b-a:.1f}s")

