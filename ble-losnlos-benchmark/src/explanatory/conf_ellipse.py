
import numpy as np
from scipy.stats import chi2
from matplotlib.patches import Ellipse


from src.explanatory.robust_center_cov_2d import _robust_center_cov_2d

def _add_conf_ellipse(ax, X, q=0.80, edgecolor="k", lw=2, linestyle="-"):
    """
    Add ellipse containing ~q probability mass (Gaussian assumption) for 2D data X.
    """
    mu, S = _robust_center_cov_2d(X, keep=q)
    # eigen-decomposition for ellipse axes
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # chi-square radius for 2 DoF
    r2 = chi2.ppf(q, df=2)

    width  = 2 * np.sqrt(vals[0] * r2)
    height = 2 * np.sqrt(vals[1] * r2)
    angle  = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))

    ell = Ellipse(xy=mu, width=width, height=height, angle=angle,
                  facecolor="none", edgecolor=edgecolor, lw=lw, linestyle=linestyle)
    ax.add_patch(ell)