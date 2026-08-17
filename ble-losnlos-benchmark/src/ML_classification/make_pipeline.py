from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def make_pipeline(spec, seed, input_dim):
    """
    Build a machine-learning pipeline from a specification dictionary.

    Supported feature representations
    ---------------------------------
    raw_input:
        X -> classifier

    scaler_only:
        X -> RobustScaler -> classifier

    linear:
        X -> RobustScaler -> PCA -> classifier

    rff:
        X -> RobustScaler -> RBFSampler -> classifier

    rbf / sigmoid / polynomial / cosine:
        X -> RobustScaler -> Nystroem -> classifier

    For SGD and MLP classifiers, StandardScaler is added after
    the feature transformation.

    Parameters
    ----------
    spec : dict
        Pipeline configuration.

    seed : int
        Random seed.

    input_dim : int
        Number of input features.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Configured pipeline.
    """

    # ---------------------------------------------------------
    # Read configuration
    # ---------------------------------------------------------
    kernel = spec["kernel"].lower()
    clf_type = spec.get("clf", "svc").lower()

    # =========================================================
    # 1. CLASSIFIER
    # =========================================================

    if clf_type == "svc":

        # NOTE:
        # This is currently an SGD logistic classifier,
        # not sklearn.svm.SVC.
        clf = SGDClassifier(
            loss="log_loss",
            alpha=float(spec.get("alpha", 1e-3)),
            penalty=spec.get("penalty", "l2"),
            max_iter=int(spec.get("max_iter", 2000)),
            tol=float(spec.get("tol", 1e-6)),
            random_state=seed,
        )

    elif clf_type == "rf":

        clf = RandomForestClassifier(
            n_estimators=int(spec.get("n_estimators", 100)),
            max_depth=spec.get("max_depth", 5),
            max_features=spec.get("max_features", "sqrt"),
            min_samples_leaf=int(
                spec.get("min_samples_leaf", 1)
            ),
            n_jobs=-1,
            random_state=seed,
        )

    elif clf_type == "mlp":

        clf = MLPClassifier(
            hidden_layer_sizes=tuple(
                spec.get(
                    "hidden_layer_sizes",
                    (32, 16),
                )
            ),
            activation=spec.get(
                "activation",
                "relu",
            ),
            alpha=float(
                spec.get(
                    "alpha",
                    1e-3,
                )
            ),
            learning_rate_init=float(
                spec.get(
                    "lr",
                    0.01,
                )
            ),
            max_iter=int(
                spec.get(
                    "max_iter",
                    2000,
                )
            ),
            early_stopping=bool(
                spec.get(
                    "early_stopping",
                    True,
                )
            ),
            random_state=seed,
        )

    else:
        raise ValueError(
            f"Unknown classifier type: {clf_type}"
        )

    # =========================================================
    # 2. RAW INPUT
    # =========================================================

    if kernel == "raw_input":
        return Pipeline([
            ("clf", clf)
        ])

    # =========================================================
    # 3. INPUT SCALER
    # =========================================================

    steps = [
        (
            "z_in",
            RobustScaler(
                with_centering=True,
                with_scaling=True,
                quantile_range=(5.0, 95.0),
            ),
        )
    ]

    # =========================================================
    # 4. FEATURE REPRESENTATION
    # =========================================================

    if kernel == "scaler_only":
        pass

    # ---------------------------------------------------------
    # Random Fourier Features
    # ---------------------------------------------------------
    elif kernel == "rff":

        steps.append(
            (
                "rff",
                RBFSampler(
                    gamma=float(spec["gamma"]),
                    n_components=int(
                        spec["n_components"]
                    ),
                    random_state=seed,
                ),
            )
        )

    # ---------------------------------------------------------
    # Linear PCA
    # ---------------------------------------------------------
    elif kernel == "linear":

        n_pca = min(
            int(spec["n_pca"]),
            input_dim,
        )

        steps.append(
            (
                "linear_pca",
                PCA(
                    n_components=n_pca,
                    random_state=seed,
                ),
            )
        )

    # ---------------------------------------------------------
    # Nystroem kernel approximation
    # RBF / Polynomial / Sigmoid / Cosine
    # ---------------------------------------------------------
    elif kernel in (
        "rbf",
        "polynomial",
        "sigmoid",
        "cosine",
    ):

        steps.append(
            (
                "nys",
                Nystroem(
                    kernel=kernel,
                    gamma=(
                        None
                        if spec.get("gamma") is None
                        else float(spec["gamma"])
                    ),
                    degree=spec.get(
                        "degree",
                        None,
                    ),
                    coef0=spec.get(
                        "coef0",
                        None,
                    ),
                    n_components=int(
                        spec["n_components"]
                    ),
                    random_state=seed,
                ),
            )
        )

    else:
        raise ValueError(
            f"Unknown kernel/feature type: {kernel}"
        )

    # =========================================================
    # 5. OUTPUT SCALING
    # =========================================================

    if clf_type in ("svc", "mlp"):

        steps.append(
            (
                "z_out",
                StandardScaler(
                    with_mean=True,
                    with_std=True,
                ),
            )
        )

    # =========================================================
    # 6. CLASSIFIER
    # =========================================================

    steps.append(
        ("clf", clf)
    )

    return Pipeline(steps)