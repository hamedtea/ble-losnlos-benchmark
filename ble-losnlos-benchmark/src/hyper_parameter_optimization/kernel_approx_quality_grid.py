import time
import numpy as np
import pandas as pd

from sklearn.kernel_approximation import RBFSampler, Nystroem
from src.hyper_parameter_optimization.get_median_gamma import median_gamma
from src.hyper_parameter_optimization.true_kernel import true_kernel
from src.hyper_parameter_optimization.alignment import alignment

def kernel_approx_quality_grid(
    X: np.ndarray,
    kernel_specs,
    components_grid,
    sample_size: int = 1000,
    random_state: int = 0,
    verbose: bool = True,
    print_every: int = 1,
) -> pd.DataFrame:

    X = np.asarray(X)
    n = X.shape[0]
    rng = np.random.default_rng(random_state)

    # ---------------------------------------------------------
    # Subsample data for approximation-quality evaluation
    # ---------------------------------------------------------
    m = min(sample_size, n)

    idx = (
        rng.choice(n, size=m, replace=False)
        if m < n
        else np.arange(n)
    )

    Xs = X[idx]

    # ---------------------------------------------------------
    # Count total iterations for progress/ETA
    # ---------------------------------------------------------
    gamma_value, sigma = median_gamma(X)

    gamma_grid = [
        gamma_value / 10,
        gamma_value,
        gamma_value * 10
    ]
    def _num_gammas_for_spec(spec):
        kern = spec.get("kernel", "rbf").lower()
        fixed_gamma = spec.get("gamma", None)

        if kern == "cosine":
            return 1

        return (
            1
            if fixed_gamma is not None
            else len(gamma_grid)
        )

    total_iters = 0

    for spec in kernel_specs:

        name = spec.get("name", spec["kernel"])
        method = spec.get(
            "method",
            "nystroem"
        ).lower()

        kern = spec.get(
            "kernel",
            "rbf"
        ).lower()

        ng = _num_gammas_for_spec(spec)
        nc = len(components_grid)

        # RFF is only valid for RBF
        if method == "rff" and kern != "rbf":
            continue

        total_iters += ng * nc

    # ---------------------------------------------------------
    # Main grid search
    # ---------------------------------------------------------
    rows = []
    it = 0

    t_global0 = time.perf_counter()

    for spec in kernel_specs:

        name = spec.get(
            "name",
            spec["kernel"]
        )

        method = spec.get(
            "method",
            "nystroem"
        ).lower()

        kern = spec.get(
            "kernel",
            "rbf"
        ).lower()

        fixed_gamma = spec.get("gamma", None)

        use_gammas = (
            [fixed_gamma]
            if fixed_gamma is not None
            else list(gamma_grid)
        )

        # Cosine kernel does not use gamma
        if kern == "cosine":
            use_gammas = [None]

        # Skip invalid RFF combinations
        if method == "rff" and kern != "rbf":
            continue

        # -----------------------------------------------------
        # Gamma loop
        # -----------------------------------------------------
        for gamma in use_gammas:

            # -------------------------------------------------
            # True kernel matrix
            # -------------------------------------------------
            if kern == "cosine":

                K_true = true_kernel(
                    Xs,
                    kern
                ).astype(
                    np.float64,
                    copy=False
                )

            else:

                K_true = true_kernel(
                    Xs,
                    kern,
                    gamma=gamma,
                    degree=spec.get("degree", None),
                    coef0=spec.get("coef0", None),
                ).astype(
                    np.float64,
                    copy=False
                )

            K_F = np.linalg.norm(
                K_true,
                "fro"
            )

            # -------------------------------------------------
            # Number of components loop
            # -------------------------------------------------
            for nc in components_grid:

                t0 = time.perf_counter()

                # -------------------------------------------------
                # Construct approximate feature map
                # -------------------------------------------------
                if method == "rff":

                    mapper = RBFSampler(
                        gamma=float(gamma),
                        n_components=int(nc),
                        random_state=random_state,
                    )

                    Phi = mapper.fit_transform(Xs)

                elif method == "nystroem":

                    mapper = Nystroem(
                        kernel=kern,
                        gamma=(
                            None
                            if gamma is None
                            else float(gamma)
                        ),
                        degree=spec.get(
                            "degree",
                            None
                        ),
                        coef0=spec.get(
                            "coef0",
                            None
                        ),
                        n_components=int(nc),
                        random_state=random_state,
                    )

                    Phi = mapper.fit_transform(Xs)

                else:
                    raise ValueError(
                        f"Unknown method: {method}"
                    )

                # -------------------------------------------------
                # Reconstruct approximate kernel
                # -------------------------------------------------
                K_hat = (
                    Phi @ Phi.T
                ).astype(
                    np.float64,
                    copy=False
                )

                iter_time = (
                    time.perf_counter() - t0
                )

                # -------------------------------------------------
                # Approximation metrics
                # -------------------------------------------------
                diffF = np.linalg.norm(
                    K_true - K_hat,
                    "fro"
                )

                relF = (
                    float(diffF / K_F)
                    if K_F > 0
                    else np.nan
                )

                R2 = (
                    float(
                        1.0
                        - (diffF ** 2)
                        / (K_F ** 2)
                    )
                    if K_F > 0
                    else np.nan
                )

                align = alignment(
                    K_true,
                    K_hat,
                    center=True
                )

                # -------------------------------------------------
                # Store result
                # -------------------------------------------------
                rows.append({
                    "name": name,
                    "method": method,
                    "kernel": kern,
                    "gamma": (
                        np.nan
                        if gamma is None
                        else float(gamma)
                    ),
                    "n_components": int(nc),
                    "alignment": align,
                    "R2_like": R2,
                    "rel_frob_error": relF,
                    "iter_time_s": iter_time,
                })

                # -------------------------------------------------
                # Progress information
                # -------------------------------------------------
                it += 1

                if verbose and (
                    it % print_every == 0
                ):

                    elapsed = (
                        time.perf_counter()
                        - t_global0
                    )

                    avg_it = (
                        elapsed / it
                    )

                    eta = (
                        avg_it
                        * (total_iters - it)
                    )

                    g_str = (
                        "None"
                        if gamma is None
                        else f"{float(gamma):.3g}"
                    )

                    # Optional progress print:
                    #
                    # print(
                    #     f"[{it:4d}/{total_iters}] "
                    #     f"{name:14s} "
                    #     f"gamma={g_str:>8s} "
                    #     f"nc={int(nc):4d} | "
                    #     f"iter={iter_time:6.3f}s "
                    #     f"elapsed={elapsed:7.1f}s "
                    #     f"ETA={eta:7.1f}s"
                    # )

    # ---------------------------------------------------------
    # Return results sorted
    # ---------------------------------------------------------
    df = (
        pd.DataFrame(rows)
        .sort_values(
            ["kernel", "gamma", "n_components"]
        )
        .reset_index(drop=True)
    )

    return df