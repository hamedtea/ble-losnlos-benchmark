from src.plotters.plot_performance_metric import plot_performance_metric

def plot_all_performance_metrics(
    summary_results_all,
):
    """
    Plot all classification performance metrics.
    """

    metrics = {
        "Accuracy": "Accuracy",
        "Precision": "Precision",
        "Recall": "Recall",
        "F1": "F1 score",
        "TPR": "True Positive Rate",
        "TNR": "True Negative Rate",
        "kappa": r"Cohen's $\kappa$",
        "AUC": "AUC",
    }

    figures = {}

    for metric, ylabel in metrics.items():
        print(
            f"Generating Lilipot plot: {metric}"
        )

        fig, ax = plot_performance_metric(
            summary_results_all = summary_results_all,
            metric=metric,
            ylabel=ylabel,
            show_values=False,
        )

        figures[metric] = {
            "fig": fig,
            "ax": ax,
        }


    return figures