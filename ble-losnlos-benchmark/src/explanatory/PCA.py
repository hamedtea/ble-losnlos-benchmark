
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

def _pca8(X, n_components=8, random_state=0):
    """Standardize + PCA -> (N, n_components)."""
    #X = _to_numpy_matrix(X)
    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)
    pca = PCA(n_components=n_components, random_state=random_state).fit(Xz)
    Z = pca.transform(Xz)
    return Z