import numpy as np

# --- Hasler-Süsstrunk colorfulness metric (robust to MRI/noise) ---
def colorfulness_score(img_rgb: np.ndarray) -> float:
    R, G, B = img_rgb[..., 0].astype(np.float32), img_rgb[..., 1].astype(np.float32), img_rgb[..., 2].astype(np.float32)
    rg = np.abs(R - G)
    yb = np.abs(0.5*(R + G) - B)
    std_rg, std_yb = rg.std(), yb.std()
    mean_rg, mean_yb = rg.mean(), yb.mean()
    return np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)

# --- Global Shannon entropy for grayscale medical validation ---
def shannon_entropy_gray(img_gray: np.ndarray) -> float:
    hist = np.bincount(img_gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def basic_visual_checks(img_rgb: np.ndarray):
    h, w, _ = img_rgb.shape
    if h < 160 or w < 160:
        return False, "Image too small for MRI analysis"
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
    col = colorfulness_score(img_rgb)
    if col > 20.0:   # Threshold: tune with MRI validation images
        return False, "Image too colorful for MRI"
    gray = (0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]).astype(np.uint8)
    ent = shannon_entropy_gray(gray)
    if ent < 3.0 or ent > 7.8:  # Threshold: tune with MRI validation images
        return False, "Image entropy atypical for MRI scan"
    return True, {"colorfulness": col, "entropy": ent}

# --- Mahalanobis OOD gate for penultimate-layer features ---
class MahalanobisGate:
    def __init__(self, mean, cov_inv, thr):
        self.mean = mean.astype(np.float32)
        self.cov_inv = cov_inv.astype(np.float32)
        self.thr = float(thr)
    def distance(self, z: np.ndarray) -> float:
        d = z.astype(np.float32) - self.mean
        return float(d @ self.cov_inv @ d)
    def is_in_distribution(self, z: np.ndarray):
        d2 = self.distance(z)
        return (d2 <= self.thr), d2
