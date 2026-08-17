from src.explanatory.hexbin import plot_hexbin
import matplotlib.pyplot as plt


def plot_hexbin_all_datasets(
    data,
    n_components=None,
    random_state=None,
    gridsize=None,
    mincnt=None,
    q_area=None,
    q_ellipse=None,
    ci=None,
    frac_marg=None,
    alpha_hex_los=None,
    alpha_hex_nlos=None,
    alpha_marg_los=None,
    alpha_marg_nlos=None,
    tick_frac=None,
    gamma_norm=None,
    ratio_mode=None,
    ratio_fmt=None,
):
    """
    Create PCA hexbin plots for:

        - Room
        - Office
        - Proportional
        - Mixed

    All figures are created first and then displayed
    simultaneously with one plt.show() call.

    Parameters
    ----------
    data : dict
        Dataset dictionary containing:

        X_los_room
        X_nlos_room

        X_los_office
        X_nlos_office

        X_los_proportional
        X_nlos_proportional

        X_los_mixed
        X_nlos_mixed

    Returns
    -------
    figures : dict
        Dictionary containing figure and axes objects
        for each dataset.
    """

    # --------------------------------------------------
    # Dataset definitions
    # --------------------------------------------------
    datasets = {
        "Room": {
            "X_los": data["X_los_room"],
            "X_nlos": data["X_nlos_room"],
        },

        "Office": {
            "X_los": data["X_los_office"],
            "X_nlos": data["X_nlos_office"],
        },

        "Proportional": {
            "X_los": data["X_los_proportional"],
            "X_nlos": data["X_nlos_proportional"],
        },

        "Mixed": {
            "X_los": data["X_los_mixed"],
            "X_nlos": data["X_nlos_mixed"],
        },
    }

    figures = {}

    # --------------------------------------------------
    # Generate all four figures
    # --------------------------------------------------
    for dataset_name, dataset in datasets.items():

        print(
            f"Generating hexbin plot: {dataset_name}"
        )

        fig, axes = plot_hexbin(
            X_los=dataset["X_los"],
            X_nlos=dataset["X_nlos"],
            title=f"{dataset_name}: LOS vs NLOS",

            n_components=n_components,
            random_state=random_state,

            gridsize=gridsize,
            mincnt=mincnt,

            q_area=q_area,
            q_ellipse=q_ellipse,

            ci=ci,
            frac_marg=frac_marg,

            alpha_hex_los=alpha_hex_los,
            alpha_hex_nlos=alpha_hex_nlos,

            alpha_marg_los=alpha_marg_los,
            alpha_marg_nlos=alpha_marg_nlos,

            tick_frac=tick_frac,
            gamma_norm=gamma_norm,

            ratio_mode=ratio_mode,
            ratio_fmt=ratio_fmt,
        )

        figures[dataset_name] = {
            "fig": fig,
            "axes": axes,
        }

    return figures