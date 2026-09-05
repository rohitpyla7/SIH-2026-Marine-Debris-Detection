# models/ — SIH26057 Model Directory

## Expected model file

Place the trained YOLOv8-Nano weights here:

```
models/best.pt
```

## Model status

| Status | Condition |
|---|---|
| 🟢 REAL MODE | `models/best.pt` found and loadable |
| 🟡 DEMO MODE | File not found — falls back to rule-based demo |

The app automatically detects which mode is active at startup.
**Never pretend the model is trained when it is not.**

---

## How to get a trained model

### Option 1 — Train on SubPipeMini2 (recommended)

```bash
# 1. Download SubPipeMini2 from Zenodo
# 2. Inspect the dataset first (ALWAYS — do not guess class names)
python scripts/inspect_subpipe.py --path D:\SubPipe

# 3. Train
python train.py --data D:\SubPipe\data.yaml --epochs 100 --imgsz 640 --batch 8

# 4. Weights auto-copied to models/best.pt
# 5. Restart app — auto-switches to REAL MODE
```

### Option 2 — Train on synthetic demo data (structural test only)

```bash
python train.py --epochs 20
```

**WARNING:** Training on synthetic data produces a model that will NOT generalize
to real sonar imagery. Use only for pipeline verification.

---

## SubPipe dataset notes

- **Source:** https://github.com/remaro-network/SubPipe-dataset
- **Classes:** `Pipeline` — ONE class only. NOT marine debris.
- **HF images:** 5,030 (5000×500 px), 3,172 annotations
- **LF images:** 5,000 (2500×500 px), 3,163 annotations
- **Split:** by mission chunk — NEVER random split of sequential frames

---

## Evaluation

After training, evaluate on held-out test split:

```bash
python evaluate.py --model models/best.pt --data D:\SubPipe\data.yaml --split test
```

Results saved to `reports/evaluation_results.json`.

**Never fabricate or report metrics without specifying the test conditions.**
