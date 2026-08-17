import numpy as np
from sklearn.preprocessing import RobustScaler


def scale_training_data(
    X_tr,
    robust_quantiles=(5, 95),
):
    """
    Fit RobustScaler on the training data and return
    the scaled training data.

    Parameters
    ----------
    X_tr : array-like, shape (n_samples, n_features)
        Training feature matrix.
    robust_quantiles : tuple(int, int)
        Quantile range used by RobustScaler.

    Returns
    -------
    X_trs : np.ndarray
        Scaled training data.
    scaler : RobustScaler
        Fitted scaler for transforming future data.
    """

    X_tr = np.asarray(X_tr)

    if X_tr.ndim != 2:
        raise ValueError(
            f"X_tr must be 2D; got shape {X_tr.shape}"
        )

    q_low, q_high = robust_quantiles

    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(q_low, q_high),
    )

    X_trs = scaler.fit_transform(X_tr)

    return X_trs, scaler


def add_scaled_training_data(
    data_3k,
    data_7k,
    robust_quantiles=(5, 95),
):
    """
    Fit a RobustScaler on each training dataset independently and
    add the scaled training data to data_3k and data_7k.

    New keys:
        data_3k["Xtrs_room"]
        data_3k["Xtrs_office"]
        data_7k["Xtrs_room"]
        data_7k["Xtrs_office"]

    Also stores the fitted scalers:
        data_3k["scaler_room"]
        data_3k["scaler_office"]
        data_7k["scaler_room"]
        data_7k["scaler_office"]
    """

    for data in [data_3k, data_7k]:

        for env in ["room", "office", "mixed", "proportional"]:

            Xtr = np.asarray(data[f"Xtr_{env}"])

            scaler = RobustScaler(
                with_centering=True,
                with_scaling=True,
                quantile_range=robust_quantiles,
            )

            Xtrs = scaler.fit_transform(Xtr)

            # Add new scaled training-data key
            data[f"Xtrs_{env}"] = Xtrs

            # Keep scaler for future test/inference transformation
            data[f"scaler_{env}"] = scaler

    return data_3k, data_7k