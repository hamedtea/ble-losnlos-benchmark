import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

def report_scaling_checks(Xtr, robust_quantiles=(5, 95), outlier_thresh=3.0, ddof=1):
    """
    Fit StandardScaler and RobustScaler on TRAIN ONLY, then report:
      (1) Global checks (centering/scaling)
      (2) Outlier sensitivity (fraction |value| > outlier_thresh)

    Parameters
    ----------
    Xtr : array-like, shape (n_samples, n_features)
        Training matrix (2D).
    robust_quantiles : tuple(int,int)
        Quantile range (q_low, q_high) used by RobustScaler.
        Use (25,75) for canonical IQR; (5,95) for 90% range.
    outlier_thresh : float
        Threshold tau for outlier fraction check.
    ddof : int
        ddof for std computation (1 = sample std).
    """
    Xtr = np.asarray(Xtr)
    if Xtr.ndim != 2:
        raise ValueError(f"Xtr must be 2D; got shape {Xtr.shape}")

    q_low, q_high = robust_quantiles
    if not (0 <= q_low < q_high <= 100):
        raise ValueError("robust_quantiles must satisfy 0 <= q_low < q_high <= 100")

    # ---------- fit on TRAIN ONLY ----------
    std_scaler = StandardScaler(with_mean=True, with_std=True)
    rob_scaler = RobustScaler(with_centering=True, with_scaling=True,
                              quantile_range=(q_low, q_high))

    X_std = std_scaler.fit_transform(Xtr)
    X_rob = rob_scaler.fit_transform(Xtr)

    # ---------- helpers ----------
    def col_mean(a):   return a.mean(axis=0)
    def col_std(a):    return a.std(axis=0, ddof=ddof)
    def col_median(a): return np.median(a, axis=0)

    def col_qrng(a, p_low, p_high):
        q = np.percentile(a, [p_low, p_high], axis=0)
        return q[1] - q[0]

    # ---------- global checks ----------
    std_mean_after = col_mean(X_std)
    std_std_after  = col_std(X_std)

    rob_med_after  = col_median(X_rob)
    rob_qrng_after = col_qrng(X_rob, q_low, q_high)
    rob_iqr_after  = col_qrng(X_rob, 25, 75)  # canonical IQR (informative if quantiles != (25,75))

    # ---------- outlier sensitivity ----------
    frac_gt_std = (np.abs(X_std) > outlier_thresh).mean(axis=0)
    frac_gt_rob = (np.abs(X_rob) > outlier_thresh).mean(axis=0)

    # ---------- report ----------
    print("\n[GLOBAL checks]")
    print(f"- StandardScaler: mean(|mean_j|) = {np.mean(np.abs(std_mean_after)):.3e}, "
          f"mean(std_j) = {np.mean(std_std_after):.4f} (target ~1)")

    print(f"- RobustScaler (quantiles=({q_low},{q_high})): mean(|median_j|) = {np.mean(np.abs(rob_med_after)):.3e}, "
          f"mean(Q{q_high}-Q{q_low})_j = {np.mean(rob_qrng_after):.4f} (target ~1)")

    print(f"- RobustScaler: (extra) mean(IQR_j=Q75-Q25) = {np.mean(rob_iqr_after):.4f}")

    print("\n[OUTLIER sensitivity]")
    print(f"- mean frac(|std|>{outlier_thresh:.1f}) across features: {frac_gt_std.mean():.4f}")
    print(f"- mean frac(|rob|>{outlier_thresh:.1f}) across features: {frac_gt_rob.mean():.4f}")

    # return useful objects if you want to use them downstream
    return {
        "X_original": Xtr,
        "X_std": X_std,
        "X_rob": X_rob,
        "std_scaler": std_scaler,
        "rob_scaler": rob_scaler,
        "frac_gt_std": frac_gt_std,
        "frac_gt_rob": frac_gt_rob,
    }

def report_scaling_checks_all(
    data_3k,
    data_7k,
    robust_quantiles=(5, 95),
    outlier_thresh=1.28,
    ddof=1,
):
    """
    Run scaling checks for Room, Office, Mixed, and Proportional
    datasets at 3K and 7K sizes.

    Returns
    -------
    results : dict
        Nested dictionary:
        results["3k"]["Room"]
        results["3k"]["Office"]
        results["3k"]["Mixed"]
        results["3k"]["Proportional"]
        results["7k"]["Room"]
        ...
    """

    results = {
        "3k": {},
        "7k": {},
    }

    # =========================================================
    # 3K
    # =========================================================
    print("\n" + "=" * 70)
    print("3K DATASETS")
    print("=" * 70)

    X_room_3k = data_3k["X_room"]
    X_office_3k = data_3k["X_office"]
    X_mixed_3k = data_3k["X_mixed"]
    X_proportional_3k = data_3k["X_proportional"]


    print("\n--- Room 3K ---")
    results["3k"]["Room"] = report_scaling_checks(
        X_room_3k,
        robust_quantiles=robust_quantiles,
        outlier_thresh=outlier_thresh,
        ddof=ddof,
    )

    print("\n--- Office 3K ---")
    results["3k"]["Office"] = report_scaling_checks(
        X_office_3k,
        robust_quantiles=robust_quantiles,
        outlier_thresh=outlier_thresh,
        ddof=ddof,
    )

    print("\n--- Mixed 3K ---")
    results["3k"]["Mixed"] = report_scaling_checks(
        X_mixed_3k,
        robust_quantiles=robust_quantiles,
        outlier_thresh=outlier_thresh,
        ddof=ddof,
    )

    print("\n--- Proportional 3K ---")
    results["3k"]["Proportional"] = report_scaling_checks(
        X_proportional_3k,
        robust_quantiles=robust_quantiles,
        outlier_thresh=outlier_thresh,
        ddof=ddof,
    )

    # =========================================================
    # 7K
    # =========================================================
    print("\n" + "=" * 70)
    print("7K DATASETS")
    print("=" * 70)

    X_room_7k = data_7k["X_room"]
    X_office_7k = data_7k["X_office"]
    X_mixed_7k = data_7k["X_mixed"]
    X_proportional_7k = data_7k["X_proportional"]

    print("\n--- Room 7K ---")
    results["7k"]["Room"] = report_scaling_checks(
        X_room_7k,
        robust_quantiles=robust_quantiles,
        outlier_thresh=outlier_thresh,
        ddof=ddof,
    )

    print("\n--- Office 7K ---")
    results["7k"]["Office"] = report_scaling_checks(
        X_office_7k,
        robust_quantiles=robust_quantiles,
        outlier_thresh=outlier_thresh,
        ddof=ddof,
    )

    print("\n--- Mixed 7K ---")
    results["7k"]["Mixed"] = report_scaling_checks(
        X_mixed_7k,
        robust_quantiles=robust_quantiles,
        outlier_thresh=outlier_thresh,
        ddof=ddof,
    )

    print("\n--- Proportional 7K ---")
    results["7k"]["Proportional"] = report_scaling_checks(
        X_proportional_7k,
        robust_quantiles=robust_quantiles,
        outlier_thresh=outlier_thresh,
        ddof=ddof,
    )

    return results