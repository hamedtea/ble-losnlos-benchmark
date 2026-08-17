from src.plotters.plot_timing_performance import plot_timing_metric

def plot_all_timing_metrics(
    summary_results_all,
    show=True,
):
    """
    Plot training and inference times.
    """
    print(
            f"Generating Lilipot plot for timing: training"
        )
    figures = {}

    figures["training"] = plot_timing_metric(
        summary_results_all=summary_results_all,
        metric="fit_time_s",
        ylabel="Training time (ms)",
        to_ms=True,
        log_scale=True,
    )

    print(
            f"Generating Lilipot plot for timing: inference"
        )
    figures["inference"] = plot_timing_metric(
        summary_results_all=summary_results_all,
        metric="pred_time_s",
        ylabel="Inference time (ms)",
        to_ms=True,
        log_scale=True,
    )

    return figures