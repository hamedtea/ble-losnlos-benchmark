import numpy as np

def center_gram(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    one = np.ones((n, 1), dtype=K.dtype)
    H = np.eye(n, dtype=K.dtype) - (one @ one.T) / n
    return H @ K @ H