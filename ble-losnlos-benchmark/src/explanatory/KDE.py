import numpy as np

def kde_gaussian_fixed(x, grid, bw):
    x = np.asarray(x, float)
    grid = np.asarray(grid, float)
    u = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * u**2).mean(axis=1) / (bw * np.sqrt(2*np.pi))

def adaptive_kde_abramson(x, grid, alpha=0.5, eps=1e-12):
    x = np.asarray(x, float)
    if x.size > 5000:
        x = x[np.random.default_rng(0).choice(x.size, size=5000, replace=False)]
    grid = np.asarray(grid, float)
    n = x.size
    s = x.std(ddof=1)
    if n < 2 or s == 0:
        bw = 1e-3 if s == 0 else 1.06*s*n**(-1/5)
        return kde_gaussian_fixed(x, grid, bw)
    
    h0 = 1.06 * s * n ** (-1/5)
    f0_x = kde_gaussian_fixed(x, x, h0)
    f0_x = np.maximum(f0_x, eps)
    g = np.exp(np.mean(np.log(f0_x)))
    hi = h0 * (g / f0_x) ** alpha
    hi = np.maximum(hi, 1e-6)

    u = (grid[:, None] - x[None, :]) / hi[None, :]
    return np.mean(np.exp(-0.5 * u**2) / (hi[None, :] * np.sqrt(2*np.pi)), axis=1)
