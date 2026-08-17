import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.plotters.ABBR import add_abbr_legend, ABBR, DS_COLORS

def plot_timing_metric(
    summary_results_all,
    metric="fit_time_s",
    ylabel="Training time (ms)",
    to_ms=True,
    fs=14,
    x_gap=1.5,
    dataset_jitter=0.08,
    size_offset=0.25,
    marker_3k="o",
    marker_7k="s",
    show_values=False,
    show_legend=True,
    show_abbr_legend=True,
    log_scale=True,
    log_min=0.1,
):
    """
    Plot fit or prediction time directly from summary_results_all.

    metric:
        "fit_time_s"
        "pred_time_s"
    """

    dataset_types = [
        "room",
        "office",
        "mixed",
        "proportional",
    ]

    x = np.arange(len(ABBR)) * x_gap

    fig, ax = plt.subplots(
        figsize=(max(12, 1.6 * len(ABBR) * x_gap), 4.5)
    )

    n_ds = len(dataset_types)

    for model_idx, (abbr, model_name) in enumerate(ABBR):

        for ds_idx, dataset_type in enumerate(dataset_types):

            color = DS_COLORS[dataset_type]

            ds_offset = (
                ds_idx - (n_ds - 1) / 2
            ) * dataset_jitter

            for size, marker, side in [
                ("3k", marker_3k, -1),
                ("7k", marker_7k, +1),
            ]:

                dataset_name = f"{dataset_type}_{size}"

                if dataset_name not in summary_results_all:
                    continue

                stats = summary_results_all[
                    dataset_name
                ]["val_stats"]

                if model_name not in stats:
                    continue

                value = stats[
                    model_name
                ].get(metric, np.nan)

                if not np.isfinite(value):
                    continue

                if to_ms:
                    value *= 1000.0

                if value <= 0:
                    continue

                xi = (
                    x[model_idx]
                    + ds_offset
                    + side * size_offset
                )

                stem_bottom = (
                    log_min
                    if log_scale
                    else 0.0
                )

                ax.vlines(
                    xi,
                    stem_bottom,
                    value,
                    color=color,
                    linewidth=2,
                    alpha=0.85,
                )

                ax.plot(
                    xi,
                    value,
                    marker=marker,
                    linestyle="",
                    color=color,
                    markersize=7,
                )

                if show_values:

                    ax.annotate(
                        f"{value:.1f}",
                        (xi, value),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        fontsize=fs - 3,
                        rotation=90,
                    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [abbr for abbr, _ in ABBR],
        fontsize=fs + 2,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=fs,
    )


    if log_scale:
        ax.set_yscale("log", base=10)
        ax.set_ylim(bottom=log_min)

    ax.grid(
        axis="y",
        alpha=0.4,
    )

    if show_legend:

        dataset_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=DS_COLORS[ds],
                label=ds.upper(),
                markersize=8,
            )
            for ds in dataset_types
        ]

        size_handles = [
            Line2D(
                [0],
                [0],
                marker=marker_3k,
                linestyle="",
                color="black",
                label="3K",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                marker=marker_7k,
                linestyle="",
                color="black",
                label="7K",
                markersize=8,
            ),
        ]

        ax.legend(
            handles=dataset_handles + size_handles,
            ncol=6,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            fontsize=fs - 2,
        )

        # =====================================================
    # ABBR description box
    # =====================================================
    if show_abbr_legend:

        add_abbr_legend(
            fig,
            abbr=ABBR,
            fs=fs - 3,
            y=0.15,
        )


    # =====================================================
    # Layout
    # =====================================================
    fig.tight_layout(
        rect=[0.0, 0.20, 1.0, 0.95]
    )

    return fig, ax