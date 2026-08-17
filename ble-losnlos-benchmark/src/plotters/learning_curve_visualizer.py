# src/plotters/learning_curve_visualizer.py

import matplotlib.pyplot as plt


def learning_curve_visualizer(
    learning_results,
    title="Learning Curve Comparison",
    show_std=True,
):
    """
    Visualize learning curves for all datasets and classifiers
    in a single figure.

    Parameters
    ----------
    learning_results : dict
        Expected structure:

        {
            "room": {
                "3k": {
                    "SVC": df,
                    "RF": df,
                    "MLP": df,
                },
                "7k": {
                    "SVC": df,
                    "RF": df,
                    "MLP": df,
                },
            },

            "office": {...},
            "proportional": {...},
            "mixed": {...},
        }

        Each DataFrame must contain:
            train_size
            validation_mean
            validation_std

    title : str
        Figure title.

    show_std : bool
        If True, show ±1 standard deviation around
        the validation curve.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """

    fig, ax = plt.subplots(
        figsize=(14, 9)
    )

    # --------------------------------------------------
    # Line styles distinguish classifiers
    # --------------------------------------------------
    clf_styles = {
        "SVC": "-",
        "RF": "--",
        "MLP": "-.",
    }

    # --------------------------------------------------
    # Plot every dataset
    # --------------------------------------------------
    for dataset_name, dataset_results in learning_results.items():
        print(
            f"Generating learning curve plot: {dataset_name}"
        )
        # We mainly use the 7K curve because it covers
        # the region from small sample sizes toward 7K.
        results_7k = dataset_results["7k"]

        for clf_name, df in results_7k.items():

            x = df["train_size"].to_numpy()

            y = df[
                "validation_mean"
            ].to_numpy()

            y_std = df[
                "validation_std"
            ].to_numpy()

            label = (
                f"{dataset_name.capitalize()} "
                f"- {clf_name}"
            )

            line, = ax.plot(
                x,
                y,
                linestyle=clf_styles.get(
                    clf_name,
                    "-"
                ),
                linewidth=2,
                marker="o",
                markersize=4,
                label=label,
            )

            # ------------------------------------------
            # Standard deviation region
            # ------------------------------------------
            if show_std:

                ax.fill_between(
                    x,
                    y - y_std,
                    y + y_std,
                    alpha=0.12,
                    color=line.get_color(),
                )

    # --------------------------------------------------
    # Mark 3K region
    # --------------------------------------------------
    ax.axvline(
        x=3000,
        linestyle=":",
        linewidth=2,
        label="3K reference",
    )

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------
    ax.set_title(
        title,
        fontsize=18,
    )

    ax.set_xlabel(
        "Number of training samples",
        fontsize=14,
    )

    ax.set_ylabel(
        "Cross-validation accuracy",
        fontsize=14,
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        fontsize=10,
        ncol=2,
        loc="best",
    )

    fig.tight_layout()

    return fig, ax