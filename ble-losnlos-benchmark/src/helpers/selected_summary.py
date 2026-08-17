import pandas as pd

from src.plotters.ABBR import ABBR


def selected_pipeline_latex_table(
    df_combined,
    caption="Mean performance and computational cost of selected pipelines.",
    label="tab:selected_pipeline_results",
):
    """
    Select pipelines defined in ABBR and create a LaTeX table
    containing:

        - Accuracy
        - F1
        - Training time
        - Inference time

    Parameters
    ----------
    df_combined : pd.DataFrame
        Output from mean_std_summary(summary_results_all).

    abbr : list of tuple
        List containing:
            (abbreviation, full model name)

    caption : str
        LaTeX table caption.

    label : str
        LaTeX table label.

    Returns
    -------
    df_selected : pd.DataFrame
        Filtered table.

    latex_table : str
        LaTeX table as a string.
    """

    # =====================================================
    # Full model name -> abbreviation
    # =====================================================
    model_to_abbr = {
        full_name: short_name
        for short_name, full_name in ABBR
    }

    # =====================================================
    # Keep only selected pipelines
    # =====================================================
    df_selected = df_combined[
        df_combined["Model"].isin(model_to_abbr.keys())
    ].copy()

    # =====================================================
    # Replace full model names with abbreviations
    # =====================================================
    df_selected["Model"] = (
        df_selected["Model"]
        .map(model_to_abbr)
    )

    # =====================================================
    # Keep only required columns
    # =====================================================
    df_selected = df_selected[
        [
            "Model",
            "Accuracy",
            "F1",
            "t_fit_s",
            "t_pred_s",
        ]
    ].copy()

    # =====================================================
    # Rename columns
    # =====================================================
    df_selected = df_selected.rename(
        columns={
            "Model": "Pipeline",
            "t_fit_s": "Training time (s)",
            "t_pred_s": "Inference time (s)",
        }
    )

    # =====================================================
    # Preserve ABBR order
    # =====================================================
    pipeline_order = [
        short_name
        for short_name, _ in ABBR
    ]

    df_selected["Pipeline"] = pd.Categorical(
        df_selected["Pipeline"],
        categories=pipeline_order,
        ordered=True,
    )

    df_selected = (
        df_selected
        .sort_values("Pipeline")
        .reset_index(drop=True)
    )

    # =====================================================
    # Replace Unicode ± with LaTeX \pm
    # =====================================================
    metric_columns = [
        "Accuracy",
        "F1",
        "Training time (s)",
        "Inference time (s)",
    ]

    for col in metric_columns:

        df_selected[col] = (
            df_selected[col]
            .astype(str)
            .str.replace(
                " ± ",
                r" $\pm$ ",
                regex=False,
            )
        )

    # =====================================================
    # Generate LaTeX
    # =====================================================
    latex_table = df_selected.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        column_format="lcccc",
    )
    print("\nSelected pipeline table:")
    print(df_selected.to_string(index=False))

    print("\nLaTeX table:")
    print(latex_table)

    return df_selected, latex_table