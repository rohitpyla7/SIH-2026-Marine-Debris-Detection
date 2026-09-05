# Automated Underwater Marine Debris & Anomaly Detection System

An automated inspection framework and Streamlit dashboard for detecting underwater debris, pipelines, and acoustic anomalies using Side-Scan Sonar (SSS) imagery.

---

## Features

- **Preprocessing & Enhancement**: Contrast-Limited Adaptive Histogram Equalization (CLAHE), adaptive Gaussian/Bilateral denoising, and dynamic range normalization.
- **Acoustic Quality Screening**: Quantitative frame evaluation measuring focus sharpness (Laplacian variance), contrast standard deviation, and column dropout.
- **Object Detection**: Neural object detection powered by Ultralytics YOLOv8, with built-in heuristic fallback mode for evaluation without GPU hardware.
- **Physics-Informed Acoustic Shadow Validation**: Target elevation validation via acoustic shadow tracking relative to the central sonar nadir track.
- **Statistical Seabed Anomaly Scoring**: Spatial entropy, edge density, and intensity distribution analysis for open-set acoustic anomaly detection.
- **Multi-Modal Evidence Fusion**: Synthesis of detector confidence, shadow metrics, and regional anomaly signals into an aggregated evidence score.
- **Export & Reporting**: Comprehensive PDF survey reports (ReportLab), geospatial map exports (Folium), and structured CSV/JSON logging.

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repo-url>
cd sih

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Demo Mission Data

Generate simulation survey frames and seed the local SQLite database:

```bash
python scripts/setup_demo.py
```

### 3. Run Acceptance & Integration Tests

Verify pipeline stages, metrics computation, and report generation:

```bash
python scripts/acceptance_test.py
python scripts/integration_test.py
```

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

Access the interactive web console at **http://localhost:8501**.

---

## Repository Architecture

```
.
├── app.py                  # Streamlit mission dashboard & inspection UI
├── train.py                # YOLOv8 model training script
├── evaluate.py             # Evaluation suite (Precision, Recall, mAP50, mAP50-95)
├── inference.py            # Headless CLI inference script
├── requirements.txt        # Python dependency specification
│
├── ai/
│   ├── detection/          # YOLOv8 wrapper & heuristic fallback detector
│   ├── anomaly/            # Regional anomaly & texture entropy scoring
│   ├── shadow_analysis/    # Physics-based acoustic shadow validation
│   ├── fusion/             # Multi-signal confidence fusion engine
│   ├── preprocessing/      # Equalization and enhancement filters
│   └── tracking/           # Multi-frame association & track management
│
├── database/               # SQLAlchemy ORM models & session utilities
├── datasets/               # Dataset adapters (YOLO format & SubPipe structures)
├── services/               # Pipeline orchestration & mission database services
├── simulation/             # Synthetic sonar generator with speckle backscatter
├── utils/                  # Image quality, geolocation, and PDF report engines
└── scripts/                # Setup, acceptance, and integration test runners
```

---

## Training Custom Models

To train on a custom YOLO-format sonar dataset (e.g. SubPipe or custom marine survey datasets):

```bash
# 1. Inspect dataset structure and split integrity
python train.py --inspect-only --data path/to/data.yaml

# 2. Train YOLOv8-Nano
python train.py --data path/to/data.yaml --epochs 100 --imgsz 640 --batch 8

# 3. Best weights are saved to models/best.pt and automatically detected on restart
```

---

## License

MIT License.
