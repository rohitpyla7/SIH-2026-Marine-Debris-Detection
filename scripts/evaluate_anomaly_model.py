from pathlib import Path
import cv2
import numpy as np
from ai.anomaly.anomaly_detector import AnomalyDetector

detector = AnomalyDetector()

def scores(folder):
    values = []
    for p in sorted(Path(folder).glob("*")):
        img = cv2.imread(str(p))
        if img is not None:
            values.append(detector._ae_score(img))
    return np.array(values)

train = scores("data/anomaly_ae/train")
val = scores("data/anomaly_ae/val")

print("=== AUTOENCODER SCORE EVALUATION ===")
print(f"Train count : {len(train)}")
print(f"Train mean  : {train.mean():.6f}")
print(f"Train median: {np.median(train):.6f}")
print(f"Train max   : {train.max():.6f}")
print()
print(f"Val count   : {len(val)}")
print(f"Val mean    : {val.mean():.6f}")
print(f"Val median  : {np.median(val):.6f}")
print(f"Val max     : {val.max():.6f}")
print()
print(f"Threshold   : {detector.threshold:.6f}")
print(f"Train >= threshold: {(train >= 1.0).sum()}")
print(f"Val >= threshold  : {(val >= 1.0).sum()}")
