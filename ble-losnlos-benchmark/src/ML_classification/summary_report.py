import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
)


def summary_report(
    ml_datasets,
    model_results_all,
    trained_models_all,
    class_order=(-1, +1),
    pos_label=+1,
    top_n=30,
):
    """
    Generate and print summary reports for all datasets.

    Parameters
    ----------
    ml_datasets : dict
        Dictionary containing Xtr, Ytr, Xte, Yte
        for every dataset.

    model_results_all : dict
        Prediction results for every dataset and model.

    trained_models_all : dict
        Trained models for every dataset.

    class_order : tuple
        Class order, default (-1, +1).

    pos_label : int
        Positive class label.

    top_n : int
        Number of top models to print.

    Returns
    -------
    summary_results_all : dict
        Summary results for every dataset.
    """

    summary_results_all = {}

    # =========================================================
    # Dataset loop
    # =========================================================
    for dataset_name, data in ml_datasets.items():

        Xte = data["Xte"]
        Yte = data["Yte"]

        model_results = model_results_all[dataset_name]
        trained_models = trained_models_all[dataset_name]

        labels = []
        accs = []
        cms = []

        stats = {}
        y_scores = {}
        probas = {}

        # =====================================================
        # Model loop
        # =====================================================
        for name, result in model_results.items():

            yhat = result["yhat"]
            cm = result["cm"]

            # -------------------------------------------------
            # Confusion matrix values
            # -------------------------------------------------
            tn, fp = cm[0, 0], cm[0, 1]
            fn, tp = cm[1, 0], cm[1, 1]

            N = cm.sum()

            # -------------------------------------------------
            # Metrics
            # -------------------------------------------------
            acc = accuracy_score(
                Yte,
                yhat,
            )

            precision = precision_score(
                Yte,
                yhat,
                pos_label=pos_label,
                zero_division=0,
            )

            recall = recall_score(
                Yte,
                yhat,
                pos_label=pos_label,
                zero_division=0,
            )

            f1 = f1_score(
                Yte,
                yhat,
                pos_label=pos_label,
                zero_division=0,
            )

            tpr = (
                tp / (tp + fn)
                if (tp + fn) > 0
                else np.nan
            )

            tnr = (
                tn / (tn + fp)
                if (tn + fp) > 0
                else np.nan
            )

            kappa = cohen_kappa_score(
                Yte,
                yhat,
            )

            # -------------------------------------------------
            # Expected agreement Pe
            # -------------------------------------------------
            row_totals = cm.sum(axis=1)
            col_totals = cm.sum(axis=0)

            pe = (
                np.sum(row_totals * col_totals)
                / (N ** 2)
                if N > 0
                else np.nan
            )

            # -------------------------------------------------
            # Probability and AUC
            # -------------------------------------------------
            auc_te = np.nan

            model = trained_models[name]

            if hasattr(model, "predict_proba"):

                proba = model.predict_proba(Xte)

                probas[name] = proba.copy()

                classes = list(
                    model.named_steps["clf"].classes_
                )

                if pos_label in classes:

                    pos_idx = classes.index(pos_label)

                    y_score = proba[:, pos_idx]

                    y_scores[name] = y_score.copy()

                    auc_te = roc_auc_score(
                        Yte,
                        y_score,
                    )

            # -------------------------------------------------
            # Save result
            # -------------------------------------------------
            labels.append(name)
            accs.append(acc)

            cms.append(
                (name, cm)
            )

            stats[name] = {
                "Accuracy": float(acc),
                "Precision": float(precision),
                "Recall": float(recall),
                "F1": float(f1),
                "TPR": float(tpr),
                "TNR": float(tnr),
                "kappa": float(kappa),
                "Pe": float(pe),
                "AUC": (
                    float(auc_te)
                    if np.isfinite(auc_te)
                    else np.nan
                ),
                "fit_time_s": result.get(
                    "fit_time_s",
                    np.nan,
                ),
                "pred_time_s": result.get(
                    "pred_time_s",
                    np.nan,
                ),
            }

        # =====================================================
        # Sort by accuracy
        # =====================================================
        order = np.argsort(
            -np.asarray(accs)
        )

        # =====================================================
        # Print dataset summary
        # =====================================================
        print("\n" + "#" * 110)

        print(
            f"SUMMARY REPORT: {dataset_name}"
        )

        print("#" * 110)

        print(
            "\nModels sorted by TEST accuracy:\n"
        )

        for rank, j in enumerate(
            order[:top_n],
            start=1,
        ):

            model_name = labels[j]
            s = stats[model_name]

            print(
                f"{rank:02d}. "
                f"{model_name:25s} | "
                f"acc={s['Accuracy']:.4f}  "
                f"Prec={s['Precision']:.4f}  "
                f"Rec={s['Recall']:.4f}  "
                f"F1={s['F1']:.4f}  "
                f"TPR={s['TPR']:.4f}  "
                f"TNR={s['TNR']:.4f}  "
                f"kappa={s['kappa']:.4f}  "
                f"Pe={s['Pe']:.4f}  "
                f"AUC={s['AUC']:.4f}  "
                f"fit={s['fit_time_s']:.3f}s  "
                f"pred={s['pred_time_s']:.3f}s"
            )

        # =====================================================
        # Save summary for this dataset
        # =====================================================
        summary_results_all[dataset_name] = {
            "labels": labels,
            "accs": accs,
            "cms": cms,
            "val_stats": stats,
            "y_scores": y_scores,
            "probas": probas,
            "Yte": np.asarray(Yte).copy(),
            "models": trained_models,
        }

    return summary_results_all