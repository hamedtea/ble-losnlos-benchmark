from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np


def add_marginals(ax, x, y, bins="auto", frac=0.22, alpha=0.6, ci=0.75, tick_frac=0.12, color='orange', mode='left-right'):
    """
    Marginal histograms (top/right) + short CI ticks at the central ci interval.
    tick_frac controls tick length as a fraction of the marginal axis height/width.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    qlo = (1.0 - ci) / 2.0
    qhi = 1.0 - qlo
    x_lo, x_hi = np.quantile(x, [qlo, qhi])
    y_lo, y_hi = np.quantile(y, [qlo, qhi])

    # ---- top histogram (x) ----
    ax_top = inset_axes(ax, width="100%", height=f"{int(frac*100)}%", loc="upper center",
                        bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    ax_top.hist(x, bins=bins, alpha=alpha, color=color)
    ax_top.set_xlim(ax.get_xlim())

    # short vertical ticks for central 95% (at 2.5% and 97.5%)
    y0, y1 = ax_top.get_ylim()
    tick_h = 4*tick_frac * (y1 - y0)
    ax_top.vlines([x_lo, x_hi], y0, y0 + tick_h, linewidth=2, color=color)

    ax_top.axis("off")
    if mode=='left-right':
        # ---- right histogram (y) ----
        ax_right = inset_axes(ax, width=f"{int(frac*100)}%", height="100%", loc="center right",
                            bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        ax_right.hist(y, bins=bins, orientation="horizontal", alpha=alpha, color=color)
        ax_right.set_ylim(ax.get_ylim())

        # short horizontal ticks for central 95% (at 2.5% and 97.5%)
        x0, x1 = ax_right.get_xlim()
        tick_w = 4*tick_frac * (x1 - x0)
        ax_right.hlines([y_lo, y_hi], x0, x0 + tick_w, linewidth=2, color=color)

        ax_right.axis("off")