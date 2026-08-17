import numpy as np
from scipy.stats import chi2

from src.explanatory.robust_center_cov_2d import _robust_center_cov_2d

def conf_ellipse_area_2d(X, q=0.80):
    """
    Area of the 2D confidence ellipse containing probability mass q
    under a Gaussian model with robust (trimmed) covariance.
    """
    _, S = _robust_center_cov_2d(X, keep=q)
    r2 = chi2.ppf(q, df=2)            # chi-square radius^2
    return float(np.pi * r2 * np.sqrt(np.linalg.det(S)))