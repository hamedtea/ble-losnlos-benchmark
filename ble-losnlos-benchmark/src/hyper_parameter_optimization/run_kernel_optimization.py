from src.hyper_parameter_optimization.kernel_approx_quality_grid import (
    kernel_approx_quality_grid
)


def run_kernel_optimization(
    datasets,
    kernel_specs,
    components_grid=(18, 32, 64),
    sample_size=1000,
    random_state=432,
):
    """
    Run kernel approximation optimization for all datasets
    and extract the best parameters for each kernel.

    Parameters
    ----------
    datasets : dict
        Dictionary of datasets:
        {
            "dataset_name": X,
            ...
        }

    kernel_specs :
        Kernel specifications used by kernel_approx_quality_grid().

    components_grid : iterable
        Number of components to test.

    sample_size : int
        Maximum sample size used for approximation-quality testing.

    random_state : int
        Random seed.

    Returns
    -------
    kernel_results : dict
        Full DataFrame result for every dataset.

    best_params_all : dict
        Best parameters for every kernel in every dataset.
    """

    kernel_results = {}
    best_params_all = {}

    # =========================================================
    # Loop over datasets
    # =========================================================
    for name, X in datasets.items():

        print("\n" + "=" * 70)
        print(f"Kernel optimization for dataset: {name}")
        print("=" * 70)

        # -----------------------------------------------------
        # Run kernel approximation optimization
        # -----------------------------------------------------
        kernel_results[name] = kernel_approx_quality_grid(
            X=X,
            kernel_specs=kernel_specs,
            components_grid=list(components_grid),
            sample_size=min(sample_size, X.shape[0]),
            random_state=random_state,
        )

        df = kernel_results[name]

        # -----------------------------------------------------
        # Find best parameters for each kernel
        # -----------------------------------------------------
        best_params = {}

        for kernel_name in [
            "rbf",
            "sigmoid",
            "polynomial",
            "cosine",
        ]:

            df_kernel = df[
                df["kernel"] == kernel_name
            ]

            if df_kernel.empty:
                continue

            best_row = df_kernel.loc[
                df_kernel["rel_frob_error"].idxmin()
            ]

            best_params[kernel_name] = {
                "gamma": best_row["gamma"],
                "n_components": int(
                    best_row["n_components"]
                ),
                "rel_frob_error": float(
                    best_row["rel_frob_error"]
                ),
                "alignment": float(
                    best_row["alignment"]
                ),
                "R2_like": float(
                    best_row["R2_like"]
                ),
            }

        # -----------------------------------------------------
        # Save best parameters for this dataset
        # -----------------------------------------------------
        best_params_all[name] = best_params

        # -----------------------------------------------------
        # Print best parameters
        # -----------------------------------------------------
        print(
            f"\nBest kernel parameters for dataset: {name}"
        )

        for kernel_name, params in best_params.items():

            print(f"\n  {kernel_name.upper()}")

            print(
                f"    gamma        : "
                f"{params['gamma']}"
            )

            print(
                f"    n_components : "
                f"{params['n_components']}"
            )

            print(
                f"    rel error    : "
                f"{params['rel_frob_error']:.6f}"
            )

            print(
                f"    alignment    : "
                f"{params['alignment']:.6f}"
            )

            print(
                f"    R2-like      : "
                f"{params['R2_like']:.6f}"
            )

    return kernel_results, best_params_all