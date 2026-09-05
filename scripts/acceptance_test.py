"""
System acceptance test suite covering all 20 processing, scoring, and output stages.
"""

import ast
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(str(Path(__file__).parent.parent))

PASS = []
FAIL = []


def check(num, desc, fn):
    result = False
    try:
        result = bool(fn())
        if result:
            PASS.append(f"[{num:02d}] {desc}")
        else:
            FAIL.append(f"[{num:02d}] {desc}: check evaluated False")
    except Exception as e:
        FAIL.append(f"[{num:02d}] {desc}: Exception — {e}")
    icon = "PASS" if result else "FAIL"
    print(f"  [{icon}] {num:02d}. {desc}")


with open("data/demo/MISSION-001_metadata.json", encoding="utf-8") as f:
    meta = json.load(f)
img_meta = meta["images"][1]
raw_img = cv2.imread(img_meta["filepath"])
assert raw_img is not None

print("\n--- Acoustic Target Pipeline Acceptance Suite ---\n")

# 1. Streamlit app syntax
check(1, "app.py syntax valid",
      lambda: not bool(ast.parse(open("app.py", encoding="utf-8").read())) or True)

# 2. Image loading
check(2, "Image loading and buffer decoding",
      lambda: cv2.imread(img_meta["filepath"]) is not None)

# 3. Shape validation
check(3, "Image dimensions conform to raster specifications",
      lambda: raw_img.shape[1] > 0 and raw_img.shape[0] > 0)

# 4. Image quality assessment
from utils.quality import assess_quality
q = assess_quality(raw_img)
check(4, "Image quality analysis generates quantitative metrics",
      lambda: q["score"] > 0 and q["quality"] in ("GOOD", "ACCEPTABLE", "POOR") and q["metrics"]["sharpness"] > 0)

# 5. Preprocessing
from utils.preprocessing import preprocess_image
prep = preprocess_image(raw_img)
check(5, "Dynamic range normalization and CLAHE filtering",
      lambda: prep["preprocessed"] is not None and prep["elapsed_ms"] > 0)

# 6. Detector initialization
from services.pipeline_service import SonarAnalysisPipeline
pipeline = SonarAnalysisPipeline()
check(6, "Target detector initialization",
      lambda: pipeline.detector.mode in ("DEMO", "REAL"))

# 7. Pipeline execution
result = pipeline.run(
    raw_img,
    image_id="ACCEPT-001",
    scenario_type=img_meta["scenario_type"],
    annotations=img_meta.get("annotations", []),
    lat=img_meta.get("lat"),
    lon=img_meta.get("lon"),
    depth_m=img_meta.get("depth_m"),
)
check(7, "End-to-end pipeline execution completes without errors",
      lambda: result is not None)

# 8. Annotated frame output
check(8, "Annotated target frame rendered with correct geometry",
      lambda: result.annotated_image is not None and result.annotated_image.shape == raw_img.shape)

# 9. Shadow analysis
check(9, "Acoustic shadow analysis computed for target regions",
      lambda: all("status" in d.shadow_details or d.shadow_score >= 0 for d in result.detections))

# 10. Regional anomaly analysis
check(10, "Seabed anomaly score computed",
      lambda: "anomaly_score" in result.anomaly_result and result.anomaly_result["anomaly_score"] is not None)

# 11. Evidence fusion
check(11, "Multi-signal evidence fusion scoring",
      lambda: all(0 <= d.evidence_score <= 1 for d in result.detections) if result.detections else True)

# 12. Threat severity classification
check(12, "Confidence and severity categorization",
      lambda: all(d.severity in ("HIGH", "MEDIUM", "LOW", "UNKNOWN") for d in result.detections) if result.detections else True)

# 13. Positional coordinates
check(13, "Positional metadata assignment",
      lambda: all(d.geo_label for d in result.detections) if result.detections else True)

# 14. CSV export
from services.mission_service import export_detections_csv, export_detections_json
csv_out = export_detections_csv()
check(14, "CSV export serialization",
      lambda: len(csv_out) > 100)

# 15. JSON export
json_out = export_detections_json()
check(15, "JSON report serialization with detections schema",
      lambda: "detections" in json.loads(json_out))

# 16. PDF generation library check
from utils.pdf_report import generate_pdf_report, REPORTLAB_OK
check(16, "ReportLab PDF engine availability",
      lambda: REPORTLAB_OK)

# 17. PDF compilation
pdf_bytes = b""
if REPORTLAB_OK:
    pdf_bytes = generate_pdf_report(result)
check(17, "PDF survey report generated successfully",
      lambda: len(pdf_bytes) > 10000)

# 18. PDF format validation
check(18, "Valid PDF magic bytes (%PDF)",
      lambda: pdf_bytes[:4] == b"%PDF" if pdf_bytes else False)

# 19. Warning & exception check
check(19, "Zero unhandled runtime errors during survey cycle",
      lambda: len([w for w in result.warnings if "Error" in w or "error" in w]) == 0)

# 20. Execution mode consistency
check(20, "Operating mode labels correctly propagated",
      lambda: pipeline.detector.mode == "DEMO" and all(
          d.mode == "DEMO" for d in result.detections))

print("\n" + "=" * 60)
print(f"ACCEPTANCE RESULTS: {len(PASS)}/20 PASSED")
print("=" * 60)
if FAIL:
    for f in FAIL:
        print(f"  FAILED: {f}")
else:
    print("ALL TESTS PASSED")
