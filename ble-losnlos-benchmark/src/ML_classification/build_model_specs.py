def build_model_specs(best_params, input_dim):
    """
    Build all feature-map + classifier configurations.

    Parameters
    ----------
    best_params : dict
        Best kernel parameters obtained from the kernel-quality search.

        Expected structure, for example:
        {
            "rbf": {
                "gamma": 0.1,
                "n_components": 128
            },
            "sigmoid": {
                "gamma": 0.01,
                "n_components": 64
            },
            "polynomial": {
                "gamma": 0.1,
                "n_components": 128
            },
            "cosine": {
                "n_components": 64
            }
        }

    input_dim : int
        Number of input features.

    Returns
    -------
    specs : list
        List of tuples:
            (model_name, specification_dictionary)
    """

    # ---------------------------------------------------------
    # Feature representations
    # ---------------------------------------------------------
    base_specs = [
        (
            "Raw input",
            {
                "kernel": "raw_input"
            },
        ),

        (
            "Scaler only",
            {
                "kernel": "scaler_only"
            },
        ),

        (
            "Linear",
            {
                "kernel": "linear",
                "n_pca": input_dim,
            },
        ),

        (
            "RBF",
            {
                "kernel": "rbf",
                "gamma": float(
                    best_params["rbf"]["gamma"]
                ),
                "n_components": int(
                    best_params["rbf"]["n_components"]
                ),
            },
        ),

        (
            "Sigmoid",
            {
                "kernel": "sigmoid",
                "gamma": float(
                    best_params["sigmoid"]["gamma"]
                ),
                "coef0": 0.0,
                "n_components": int(
                    best_params["sigmoid"]["n_components"]
                ),
            },
        ),

        (
            "Polynomial",
            {
                "kernel": "polynomial",
                "gamma": float(
                    best_params["polynomial"]["gamma"]
                ),
                "degree": 3,
                "coef0": 1.0,
                "n_components": int(
                    best_params["polynomial"]["n_components"]
                ),
            },
        ),

        (
            "Cosine",
            {
                "kernel": "cosine",
                "gamma": None,
                "n_components": int(
                    best_params["cosine"]["n_components"]
                ),
            },
        ),

        (
            "RFF",
            {
                "kernel": "rff",
                "gamma": float(
                    best_params["rbf"]["gamma"]
                ),
                "n_components": int(
                    best_params["rbf"]["n_components"]
                ),
            },
        ),
    ]

    # ---------------------------------------------------------
    # Classifiers
    # ---------------------------------------------------------
    classifier_specs = [
        (
            "SVC",
            {
                "clf": "svc"
            },
        ),

        (
            "RF",
            {
                "clf": "rf",
                "n_estimators": 100,
                "max_depth": 5,
            },
        ),

        (
            "MLP",
            {
                "clf": "mlp",
                "hidden_layer_sizes": (16, 8),
                "max_iter": 2000,
            },
        ),
    ]

    # ---------------------------------------------------------
    # Combine every feature representation with every classifier
    # ---------------------------------------------------------
    specs = []

    for feature_name, feature_spec in base_specs:

        for classifier_name, classifier_spec in classifier_specs:

            spec = feature_spec.copy()
            spec.update(classifier_spec)

            model_name = (
                f"{feature_name} + {classifier_name}"
            )

            specs.append(
                (model_name, spec)
            )

    return specs