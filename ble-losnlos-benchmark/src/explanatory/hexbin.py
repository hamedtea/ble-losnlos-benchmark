from matplotlib.colors import PowerNorm
from scipy.stats import chi2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


from src.explanatory.robust_center_cov_2d import _robust_center_cov_2d
from src.explanatory.PCA import _pca8
from src.explanatory.KDE import adaptive_kde_abramson
from src.explanatory.conf_elipse2D import conf_ellipse_area_2d
from src.explanatory.config import CFG
from src.explanatory.marginal import add_marginals
from src.explanatory.marginal import add_marginals
from src.explanatory.kurtosis import reduce_kurtosis
from src.explanatory.conf_ellipse import _add_conf_ellipse






def plot_hexbin(
    X_los, X_nlos, title=None,
    n_components=None, random_state=None,
    gridsize=None, mincnt=None,
    q_area=None, q_ellipse=None,
    ci=None, frac_marg=None,
    alpha_hex_los=None, alpha_hex_nlos=None,
    alpha_marg_los=None, alpha_marg_nlos=None,
    tick_frac=None, gamma_norm=None,
    ratio_mode=None, ratio_fmt=None,
):
    # -------- defaults from CFG --------
    n_components   = CFG["n_components"]   if n_components   is None else n_components
    random_state   = CFG["random_state"]   if random_state   is None else random_state
    gridsize       = CFG["gridsize"]       if gridsize       is None else gridsize
    mincnt         = CFG["mincnt"]         if mincnt         is None else mincnt
    q_area         = CFG["q_area"]         if q_area         is None else q_area
    q_ellipse      = CFG["q_ellipse"]      if q_ellipse      is None else q_ellipse
    ci             = CFG["ci"]             if ci             is None else ci
    frac_marg      = CFG["frac_marg"]      if frac_marg      is None else frac_marg
    alpha_hex_los  = CFG["alpha_hex_los"]  if alpha_hex_los  is None else alpha_hex_los
    alpha_hex_nlos = CFG["alpha_hex_nlos"] if alpha_hex_nlos is None else alpha_hex_nlos
    alpha_marg_los = CFG["alpha_marg_los"] if alpha_marg_los is None else alpha_marg_los
    alpha_marg_nlos= CFG["alpha_marg_nlos"]if alpha_marg_nlos is None else alpha_marg_nlos
    tick_frac      = CFG["tick_frac"]      if tick_frac      is None else tick_frac
    gamma_norm     = CFG["gamma_norm"]     if gamma_norm     is None else gamma_norm
    ratio_mode     = CFG["ratio_mode"]     if ratio_mode     is None else ratio_mode
    ratio_fmt      = CFG["ratio_fmt"]      if ratio_fmt      is None else ratio_fmt

    
    # -------- PCA (separately for LOS and NLOS, as in your original) --------
    Z1 = _pca8(X_los,  n_components=n_components, random_state=random_state)
    Z2 = _pca8(X_nlos, n_components=n_components, random_state=random_state)

    df_los  = pd.DataFrame(Z1, columns=[f"PC{i+1}" for i in range(n_components)])
    df_nlos = pd.DataFrame(Z2, columns=[f"PC{i+1}" for i in range(n_components)])

    X1 = df_los.to_numpy()
    X2 = df_nlos.to_numpy()
    cols = df_los.columns
    k = df_los.shape[1]

    # -------- precompute ellipse areas for ALL (i,j) --------
    los_area  = {}
    nlos_area = {}
    for i in range(k):
        for j in range(k):
            Xpair_los  = np.c_[X1[:, j], X1[:, i]]
            Xpair_nlos = np.c_[X2[:, j], X2[:, i]]
            los_area[(i, j)]  = conf_ellipse_area_2d(Xpair_los,  q=q_area)
            nlos_area[(i, j)] = conf_ellipse_area_2d(Xpair_nlos, q=q_area)

    # -------- plot grid --------
    fig, axes = plt.subplots(k, k, figsize=(12, 12), constrained_layout=True)
    fig.suptitle(title)

    for i in range(k):
        for j in range(k):
            ax = axes[i, j]

            if i == j:
                xg = np.linspace(
                    min(X1[:, i].min(), X2[:, i].min()),
                    max(X1[:, i].max(), X2[:, i].max()),
                    400
                )
                y1 = adaptive_kde_abramson(X1[:, i], xg, alpha=0.5)
                y2 = adaptive_kde_abramson(X2[:, i], xg, alpha=0.5)
                ax.plot(xg, y1, label="LOS")
                ax.plot(xg, y2, label="NLOS")
                add_marginals(ax, X1[:, i], X1[:, i], bins="auto", frac=frac_marg, alpha=alpha_marg_los,
                              ci=ci, tick_frac=tick_frac, color="blue", mode='top')
                add_marginals(ax, X2[:, i], X2[:, i], bins="auto", frac=frac_marg, alpha=alpha_marg_los,
                              ci=ci, tick_frac=tick_frac, color="orange", mode='top')
                #ax.set_yticks([])
            

            elif j < i:
                # ----- LOS lower triangle -----
                x = X1[:, j]; y = X1[:, i]
                ax.hexbin(
                    x, y, gridsize=gridsize,
                    reduce_C_function=reduce_kurtosis,
                    mincnt=mincnt, alpha=alpha_hex_los,
                    cmap="Blues", norm=PowerNorm(gamma=gamma_norm)
                )
                add_marginals(ax, x, y, bins="auto", frac=frac_marg, alpha=alpha_marg_los,
                              ci=ci, tick_frac=tick_frac, color="blue")
                _add_conf_ellipse(ax, np.c_[x, y], q=q_ellipse, edgecolor="tab:blue", lw=2)

                aL = los_area[(i, j)]
                aN = nlos_area[(i, j)]
                ratio = (aN / aL) if ratio_mode == "nlos_over_los" else (aL / aN)
                ax.text(0.02, 1.01, f"A-ratio={ratio_fmt.format(ratio)}",
                        transform=ax.transAxes, va="bottom", fontsize=13)

            else:
                # ----- NLOS upper triangle -----
                x = X2[:, j]; y = X2[:, i]
                ax.hexbin(
                    x, y, gridsize=gridsize,
                    reduce_C_function=reduce_kurtosis,
                    mincnt=mincnt, alpha=alpha_hex_nlos,
                    cmap="Oranges", norm=PowerNorm(gamma=gamma_norm)
                )
                add_marginals(ax, x, y, bins="auto", frac=frac_marg, alpha=alpha_marg_nlos,
                              ci=ci, tick_frac=tick_frac, color="orange")
                _add_conf_ellipse(ax, np.c_[x, y], q=q_ellipse, edgecolor="tab:orange", lw=2)

# show tick numbers on: TOP row + BOTTOM row, and LEFT column
# (hide them everywhere else)

            # # ----- x ticks -----
            # if i == 0:
            #     ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
            #     ax.xaxis.set_ticks_position("top")
            #     ax.xaxis.set_label_position("top")
            #     ax.set_xlabel(cols[j])  # optional: show labels on top row
            if i == k - 1:
                ax.tick_params(axis="x", bottom=True, labelbottom=True, top=False, labeltop=False, labelsize=18)
                ax.xaxis.set_ticks_position("bottom")
                ax.xaxis.set_label_position("bottom")
                ax.set_xlabel(cols[j], fontsize=22)
            else:
                ax.set_xticks([])
                ax.tick_params(axis="x", top=False, bottom=False, labeltop=False, labelbottom=False, labelsize=18)

            # ----- y ticks -----
            if j == 0:
                ax.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False, labelsize=18)
                ax.set_ylabel(cols[i], fontsize=22)
            else:
                ax.set_yticks([])
                ax.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False, labelsize=18)
            
            if (i==0) and (j==0): 
                ax.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False, labelsize=18)
                ax.set_ylabel(cols[j], fontsize=22)


    # legends
    leg_classes = fig.legend(
        handles=[Patch(facecolor="tab:blue"), Patch(facecolor="tab:orange")],
        labels=["LOS", "NLOS"], ncols=2,
        loc="upper right", bbox_to_anchor=(0.4, 1.04), frameon=True, fontsize=18
    )
    leg_kurt = fig.legend(
        handles=[Line2D([0], [0], color="black", lw=0)],
        labels=[r"metrics: kurtosis + AKDE + marginals + ellipse"],
        loc="upper right", bbox_to_anchor=(1.03, 1.04), frameon=True, fontsize=18
    )
    fig.add_artist(leg_classes)
    fig.add_artist(leg_kurt)
    return fig, axes