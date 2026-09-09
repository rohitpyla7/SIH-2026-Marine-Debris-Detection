import glob
import os
import numpy as np

files = sorted(
    glob.glob("./data/subpipe_hf_yolo/labels/train/*.txt")
    + glob.glob("./data/subpipe_hf_yolo/labels/val/*.txt")
    + glob.glob("./data/subpipe_hf_yolo/labels/test/*.txt"),
    key=lambda x: float(os.path.basename(x)[:-4])
)

print("BLOCK       BOXES   AVG_HEIGHT")
print("-" * 35)

for i in range(0, len(files), 50):
    heights = []

    for f in files[i:i+50]:
        with open(f) as file:
            for line in file:
                if line.strip():
                    heights.append(float(line.split()[4]))

    block = f"{i//50+1:02d} ({i+1:03d}-{min(i+50,len(files)):03d})"

    if heights:
        print(f"{block}   {len(heights):<7} {np.mean(heights):.3f}")
    else:
        print(f"{block}   0       N/A")
