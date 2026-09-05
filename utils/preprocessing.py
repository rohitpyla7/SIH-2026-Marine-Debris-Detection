"""
Image preprocessing routines for side-scan sonar frames.
Includes grayscale conversion, dynamic range stretching, CLAHE, and adaptive denoising.
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np

OUTPUT_DIR = Path("outputs/preprocessing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import bm3d as _bm3d
    BM3D_AVAILABLE = True
except ImportError:
    BM3D_AVAILABLE = False


def load_image(path) -> Optional[np.ndarray]:
    """Load image from path or byte buffer."""
    if isinstance(path, (str, Path)):
        return cv2.imread(str(path), cv2.IMREAD_COLOR)
    arr = np.frombuffer(path, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to single-channel grayscale if needed."""
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def normalize(img: np.ndarray) -> np.ndarray:
    """Linearly scale pixel intensities to [0, 255]."""
    img_f = img.astype(np.float32)
    lo, hi = img_f.min(), img_f.max()
    if hi - lo < 1e-6:
        return img.astype(np.uint8)
    return np.clip(((img_f - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)


def apply_clahe(gray: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def denoise(gray: np.ndarray, method: str = "auto") -> np.ndarray:
    """Denoise grayscale sonar slice using BM3D, NLM, or Gaussian filtering."""
    if method == "bm3d" or (method == "auto" and BM3D_AVAILABLE):
        try:
            import bm3d
            denoised = bm3d.bm3d(gray, sigma_psd=25 / 255)
            return np.clip(denoised * 255, 0, 255).astype(np.uint8)
        except Exception:
            pass

    if method in ("nlm", "auto"):
        try:
            return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        except Exception:
            pass

    return cv2.GaussianBlur(gray, (5, 5), 0)


def preprocess_image(img_bgr: np.ndarray,
                      clip_limit: float = 2.0,
                      tile_grid: Tuple[int, int] = (8, 8),
                      denoise_method: str = "auto",
                      save_path: Optional[str] = None) -> dict:
    """Execute complete preprocessing chain and return intermediate stages."""
    t0 = time.time()
    steps = {}

    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Empty or invalid input image")

    steps["raw"] = img_bgr.copy()

    gray = to_gray(img_bgr)
    steps["gray"] = gray.copy()

    norm = normalize(gray)
    steps["normalized"] = norm.copy()

    clahe_img = apply_clahe(norm, clip_limit, tile_grid)
    steps["clahe"] = clahe_img.copy()

    denoised = denoise(clahe_img, method=denoise_method)
    steps["denoised"] = denoised.copy()

    preprocessed_bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    steps["preprocessed"] = preprocessed_bgr

    elapsed_ms = round((time.time() - t0) * 1000, 1)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), preprocessed_bgr)

    return {
        "preprocessed": preprocessed_bgr,
        "steps": steps,
        "elapsed_ms": elapsed_ms,
        "clahe_params": {"clip_limit": clip_limit, "tile_grid": tile_grid},
        "denoise_method": "bm3d" if (denoise_method in ("bm3d", "auto") and BM3D_AVAILABLE) else "nlm_fallback",
        "bm3d_available": BM3D_AVAILABLE,
    }
