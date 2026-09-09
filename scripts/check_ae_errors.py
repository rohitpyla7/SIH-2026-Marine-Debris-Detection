import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.anomaly.anomaly_detector import AnomalyDetector
import cv2
import torch
import numpy as np

d = AnomalyDetector()
vals = []

files = list(Path("./data/anomaly_ae/val").glob("*.png"))

for p in files:
    img = cv2.imread(str(p))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    x = torch.from_numpy(
        resized.astype(np.float32) / 255.0
    ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        recon = d._ae(x)

    error = torch.mean((recon - x) ** 2).item()
    vals.append(error)

print("RAW RECONSTRUCTION ERROR")
print("IMAGES:", len(vals))
print("MIN:", round(min(vals), 6))
print("MAX:", round(max(vals), 6))
print("MEAN:", round(float(np.mean(vals)), 6))
print("MEDIAN:", round(float(np.median(vals)), 6))
print("P95:", round(float(np.percentile(vals, 95)), 6))
print("P99:", round(float(np.percentile(vals, 99)), 6))
