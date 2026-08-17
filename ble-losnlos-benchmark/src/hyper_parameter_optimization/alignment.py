
import numpy as np
from src.hyper_parameter_optimization.center_gram import center_gram

def alignment(K: np.ndarray, Kh: np.ndarray, center: bool = True) -> float:
    if center:
        K  = center_gram(K)
        Kh = center_gram(Kh)
    num = np.sum(K * Kh, dtype=np.float64)
    den = np.sqrt(np.sum(K * K, dtype=np.float64) * np.sum(Kh * Kh, dtype=np.float64))
    return float(num / den) if den > 0 else np.na