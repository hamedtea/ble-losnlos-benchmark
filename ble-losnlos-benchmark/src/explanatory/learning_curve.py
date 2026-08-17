import numpy as np
import pandas as pd

from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def get_fresh_classifiers(seed=2584):
    """
    Fresh classifiers used only for learning-curve comparison.
    """

    return {
        "SVC": Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    random_state=seed,
                ),
            ),
        ]),

        "RF": Pipeline([
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]),

        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(16, 8),
                    activation="relu",
                    alpha=1e-3,
                    learning_rate_init=0.01,
                    max_iter=2000,
                    early_stopping=True,
                    random_state=seed,
                ),
            ),
        ]),
    }

def learning_curve_comparison(
    data_3k,
    data_7k,
    dataset_type="room",
    min_size=500,
    step=500,
    cv=5,
    scoring="accuracy",
    seed=2584,
):
    """
    Compare learning curves for SVC, RF and MLP
    using the 3K and 7K versions of one dataset type.

    dataset_type:
        "room"
        "office"
        "proportional"
        "mixed"
    """

    # --------------------------------------------------
    # Get training data
    # --------------------------------------------------
    Xtr_3k = np.asarray(
        data_3k[f"Xtr_{dataset_type}"]
    )

    Ytr_3k = np.asarray(
        data_3k[f"Ytr_{dataset_type}"]
    )

    Xtr_7k = np.asarray(
        data_7k[f"Xtr_{dataset_type}"]
    )

    Ytr_7k = np.asarray(
        data_7k[f"Ytr_{dataset_type}"]
    )

    classifiers = get_fresh_classifiers(
        seed=seed
    )

    cv_split = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=seed,
    )

    results = {
        "3k": {},
        "7k": {},
    }

    # ==================================================
    # Loop over classifiers
    # ==================================================
    for clf_name, model in classifiers.items():

        # ==============================================
        # 3K
        # ==============================================
        max_train_3k = int(
            len(Xtr_3k) * (cv - 1) / cv
        )

        sizes_3k = np.arange(
            min_size,
            max_train_3k + 1,
            step,
        )

        train_sizes_3k, train_scores_3k, val_scores_3k = learning_curve(
            estimator=model,
            X=Xtr_3k,
            y=Ytr_3k,
            train_sizes=sizes_3k,
            cv=cv_split,
            scoring=scoring,
            n_jobs=-1,
            shuffle=True,
            random_state=seed,
        )

        df_3k = pd.DataFrame({
            "train_size": train_sizes_3k,
            "train_mean": np.mean(
                train_scores_3k,
                axis=1,
            ),
            "train_std": np.std(
                train_scores_3k,
                axis=1,
                ddof=1,
            ),
            "validation_mean": np.mean(
                val_scores_3k,
                axis=1,
            ),
            "validation_std": np.std(
                val_scores_3k,
                axis=1,
                ddof=1,
            ),
        })

        results["3k"][clf_name] = df_3k

        # ==============================================
        # 7K
        # ==============================================
        max_train_7k = int(
            len(Xtr_7k) * (cv - 1) / cv
        )

        sizes_7k = np.arange(
            min_size,
            max_train_7k + 1,
            step,
        )

        train_sizes_7k, train_scores_7k, val_scores_7k = learning_curve(
            estimator=model,
            X=Xtr_7k,
            y=Ytr_7k,
            train_sizes=sizes_7k,
            cv=cv_split,
            scoring=scoring,
            n_jobs=-1,
            shuffle=True,
            random_state=seed,
        )

        df_7k = pd.DataFrame({
            "train_size": train_sizes_7k,
            "train_mean": np.mean(
                train_scores_7k,
                axis=1,
            ),
            "train_std": np.std(
                train_scores_7k,
                axis=1,
                ddof=1,
            ),
            "validation_mean": np.mean(
                val_scores_7k,
                axis=1,
            ),
            "validation_std": np.std(
                val_scores_7k,
                axis=1,
                ddof=1,
            ),
        })

        results["7k"][clf_name] = df_7k

        # ==============================================
        # Print
        # ==============================================
        print("\n" + "=" * 80)
        print(
            f"{dataset_type.upper()} | {clf_name} | 3K"
        )
        print("=" * 80)

        print(
            df_3k.to_string(index=False)
        )

        print("\n" + "=" * 80)
        print(
            f"{dataset_type.upper()} | {clf_name} | 7K"
        )
        print("=" * 80)

        print(
            df_7k.to_string(index=False)
        )

    return results