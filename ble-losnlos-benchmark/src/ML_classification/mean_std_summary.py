import numpy as np
import pandas as pd


def mean_std_summary(summary_results_all):
    """
    Compute mean ± std of model metrics across all datasets.

    Parameters
    ----------
    summary_results_all : dict
        Output returned by summary_report().

        Expected structure:
        {
            "room_3k": {
                "val_stats": {
                    "RBF + RF": {
                        "Accuracy": ...,
                        "Precision": ...,
                        "Recall": ...,
                        ...
                    }
                }
            },
            ...
        }

    Returns
    -------
    df_combined : pd.DataFrame
        Mean ± std summary for every model.

    df_mean : pd.DataFrame
        Numerical mean values.

    df_std : pd.DataFrame
        Numerical standard deviations.
    """

    rows = []

    # =========================================================
    # Convert summary dictionary to DataFrame
    # =========================================================
    for dataset_name, dataset_result in summary_results_all.items():

        stats = dataset_result["val_stats"]

        for model_name, metrics in stats.items():

            fit_time = metrics.get("fit_time_s", np.nan)
            pred_time = metrics.get("pred_time_s", np.nan)

            total_time = (
                fit_time + pred_time
                if np.isfinite(fit_time) and np.isfinite(pred_time)
                else np.nan
            )

            rows.append({
                "Dataset": dataset_name,
                "Model": model_name,

                "Accuracy": metrics.get(
                    "Accuracy",
                    np.nan,
                ),

                "Precision": metrics.get(
                    "Precision",
                    np.nan,
                ),

                "Recall": metrics.get(
                    "Recall",
                    np.nan,
                ),

                "F1": metrics.get(
                    "F1",
                    np.nan,
                ),

                "TPR": metrics.get(
                    "TPR",
                    np.nan,
                ),

                "TNR": metrics.get(
                    "TNR",
                    np.nan,
                ),

                "Kappa": metrics.get(
                    "kappa",
                    np.nan,
                ),

                "AUC": metrics.get(
                    "AUC",
                    np.nan,
                ),

                "t_fit_s": fit_time,
                "t_pred_s": pred_time,
                "t_total_s": total_time,
            })

    df = pd.DataFrame(rows)

    # =========================================================
    # Metrics to summarize
    # =========================================================
    metric_cols = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "TPR",
        "TNR",
        "Kappa",
        "AUC",
        "t_fit_s",
        "t_pred_s",
        "t_total_s",
    ]

    # =========================================================
    # Mean
    # =========================================================
    df_mean = (
        df.groupby("Model")[metric_cols]
        .mean()
    )

    # =========================================================
    # Standard deviation
    # =========================================================
    df_std = (
        df.groupby("Model")[metric_cols]
        .std(ddof=1)
    )

    # =========================================================
    # Sort models by mean Kappa
    # =========================================================
    order = (
        df_mean
        .sort_values(
            by="Kappa",
            ascending=False,
        )
        .index
    )

    df_mean = df_mean.loc[order]
    df_std = df_std.loc[order]

    # =========================================================
    # Combine as mean ± std
    # =========================================================
    df_combined = pd.DataFrame(
        index=df_mean.index
    )

    for col in metric_cols:

        df_combined[col] = [
            (
                f"{mean:.3f} ± {std:.3f}"
                if pd.notna(mean) and pd.notna(std)
                else f"{mean:.3f}"
                if pd.notna(mean)
                else "--"
            )
            for mean, std in zip(
                df_mean[col],
                df_std[col],
            )
        ]

    return df_combined