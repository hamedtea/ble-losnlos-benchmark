import numpy as np

ABBR = [
    ("CS", "Cosine + SVC"),
    ("LS", "Linear + SVC"),
    ("P3S", "Polynomial + SVC"),
    ("RBS", "RBF + SVC"),
    ("RF", "Raw input + RF"),
    ("SS", "Sigmoid + SVC"),
    ("RFS", "RFF + SVC"),
    ("MLP", "Scaler only + MLP"),
]

DS_COLORS = {
    "room": "tab:blue",
    "office": "tab:orange",
    "mixed": "tab:green",
    "proportional": "tab:red",
}


def add_abbr_legend(fig, abbr=ABBR, fs=11, y=-0.02, ncol=4):
    """
    Add model abbreviation definitions below a figure.
    """

    n = len(abbr)
    nrow = int(np.ceil(n / ncol))

    cols = []

    for c in range(ncol):
        start = c * nrow
        end = min((c + 1) * nrow, n)

        cols.append([
            f"{short}: {full}"
            for short, full in abbr[start:end]
        ])

    # Pad columns
    for c in range(ncol):
        if len(cols[c]) < nrow:
            cols[c] += [""] * (
                nrow - len(cols[c])
            )

    col_widths = [
        max(len(s) for s in col)
        if col
        else 0
        for col in cols
    ]

    lines = []

    for r in range(nrow):

        parts = []

        for c in range(ncol):

            text = cols[c][r]

            parts.append(
                text.ljust(
                    col_widths[c] + 4
                )
            )

        lines.append(
            "".join(parts).rstrip()
        )

    legend_text = "\n".join(lines)

    fig.text(
        0.5,
        y,
        legend_text,
        ha="center",
        va="bottom",
        fontsize=fs,
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.95,
            linewidth=0.8,
        ),
    )