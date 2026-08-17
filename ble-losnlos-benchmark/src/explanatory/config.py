CFG = dict(
    n_components=8,
    random_state=1234,

    # hexbin
    gridsize=15,
    mincnt=10,
    alpha_hex_los=0.7,
    alpha_hex_nlos=0.7,
    gamma_norm=0.5,

    # dense-core / dispersion metrics
    q_area=0.95,
    q_ellipse=0.95,

    # marginals
    ci=0.95,
    frac_marg=0.22,
    alpha_marg_los=0.5,
    alpha_marg_nlos=0.5,
    tick_frac=0.12,

    # annotation
    ratio_mode="nlos_over_los",
    ratio_fmt="{:.2f}",
)