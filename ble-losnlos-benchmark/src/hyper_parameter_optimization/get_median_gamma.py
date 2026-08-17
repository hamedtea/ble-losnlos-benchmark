import numpy as np
from scipy.spatial.distance import pdist

def median_gamma(X, n_subsample=1000):
    if X.shape[0] > n_subsample:
        idx = np.random.choice(X.shape[0], n_subsample, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    dists = pdist(X_sample, metric='euclidean')

    sigma = np.median(dists)

    gamma = 1.0 / (2 * (sigma**2))

    return gamma, sigma