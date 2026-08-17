import numpy as np

def _robust_center_cov_2d(X, keep=0.80):
    """
    Robustly estimate (mu, Sigma) by keeping only the central keep-fraction
    according to Mahalanobis distance (iterative trimming, 1 pass).
    X: (N,2)
    """
    mu = X.mean(axis=0)
    S  = np.cov(X, rowvar=False) + 1e-9*np.eye(2)
    Sinv = np.linalg.inv(S)
    d2 = np.einsum("ni,ij,nj->n", X-mu, Sinv, X-mu)  # squared Mahalanobis
    thr = np.quantile(d2, keep)
    Xc = X[d2 <= thr]
    mu = Xc.mean(axis=0)
    S  = np.cov(Xc, rowvar=False) + 1e-9*np.eye(2)
    return mu, S