"""
Integration test validating end-to-end processing, rendering, and export serialization.
"""

import os
import sys
from pathlib import Path
import cv2
import json

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

print("Starting end-to-end integration test...\n")

with open("data/demo/MISSION-001_metadata.json", encoding="utf-8") as f:
    meta = json.load(f)
img_meta = meta["images"][1]
img = cv2.imread(img_meta["filepath"])
print(f"[1] Ingested frame: {img_meta['filename']} | shape={img.shape}")

from services.pipeline_service import SonarAnalysisPipeline
pipeline = SonarAnalysisPipeline()
result = pipeline.run(
    img,
    image_id="TEST-001",
    scenario_type=img_meta["scenario_type"],
    annotations=img_meta.get("annotations", []),
    lat=img_meta.get("lat"),
    lon=img_meta.get("lon"),
    depth_m=img_meta.get("depth_m"),
)
print(f"[2] Pipeline completed in {result.timing['total_ms']:.0f} ms | Targets: {len(result.detections)}")

q = result.quality
print(f"[3] Image quality: {q['quality']} (score: {q['score']:.2f})")

for d in result.detections:
    print(f"    Target: {d.display_name} | Conf: {d.confidence:.0%} | Evidence: {d.evidence_pct:.0f}% | Severity: {d.severity}")

ar = result.anomaly_result
print(f"[4] Seabed anomaly score: {ar['anomaly_score']:.3f}")

from services.mission_service import export_detections_csv, export_detections_json
csv_out = export_detections_csv()
json_out = export_detections_json()
print(f"[5] Exported CSV ({len(csv_out)} bytes) and JSON ({len(json_out)} bytes)")

from utils.pdf_report import generate_pdf_report
pdf_bytes = generate_pdf_report(result)
Path("results").mkdir(parents=True, exist_ok=True)
with open("results/test_report.pdf", "wb") as f:
    f.write(pdf_bytes)
print(f"[6] Rendered PDF survey report: {len(pdf_bytes):,} bytes")

cv2.imwrite("results/test_annotated.png", result.annotated_image)
print("[7] Saved test_annotated.png")

print("\nIntegration test successful.")
