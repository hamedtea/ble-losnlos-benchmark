from sklearn.metrics.pairwise import (
    rbf_kernel,
    polynomial_kernel,
    sigmoid_kernel,
    cosine_similarity,
    linear_kernel,
)


def true_kernel(X, kernel, gamma=None, degree=None, coef0=None):
    kernel = kernel.lower()

    if kernel == "rbf":
        return rbf_kernel(X, X, gamma=gamma)

    elif kernel == "polynomial":
        return polynomial_kernel(
            X,
            X,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
        )

    elif kernel == "sigmoid":
        return sigmoid_kernel(
            X,
            X,
            gamma=gamma,
            coef0=coef0,
        )

    elif kernel == "cosine":
        return cosine_similarity(X, X)

    elif kernel == "linear":
        return linear_kernel(X, X)

    else:
        raise ValueError(f"Unknown kernel: {kernel}")