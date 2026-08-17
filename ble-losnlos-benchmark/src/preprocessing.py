# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# %%
def split_los_nlos(df_room, df_office, df_mixed, df_proportional, size):
    """
    Split Room and Office DataFrames into LOS and NLOS subsets.

    Parameters
    ----------
    df_room : dict
        Dictionary containing room DataFrames, e.g.:
        {"3k": df_room_3k, "7k": df_room_7k}
    df_office : dict
        Dictionary containing office DataFrames, e.g.:
        {"3k": df_office_3k, "7k": df_office_7k}
    size : str
        Either "3k" or "7k"

    Returns
    -------
    dict
        {
            "room": {"LOS": df_LOS_room, "NLOS": df_NLOS_room},
            "office": {"LOS": df_LOS_office, "NLOS": df_NLOS_office}
        }
    """
    out = {}

    for env_name, df_dict in [("room", df_room), ("office", df_office), ("mixed", df_mixed), ("proportional", df_proportional)]:
        df = df_dict[size]
        s = df["experiment_setup"].astype(str)

        df_los = df[
            s.str.contains("LOS", case=False, na=False) &
            ~s.str.contains("NLOS", case=False, na=False)
        ].copy()

        df_nlos = df[
            s.str.contains("NLOS", case=False, na=False)
        ].copy()

        out[env_name] = {"LOS": df_los, "NLOS": df_nlos}

    return out

#split_3k = split_los_nlos(df_room, df_office, "3k")
#split_7k = split_los_nlos(df_room, df_office, "7k")

# %%
def make_xy_arrays(split_dict, feature_col="iq_64abs"):
    """
    Convert split DataFrames into NumPy feature arrays.

    Parameters
    ----------
    split_dict : dict
        Output of split_los_nlos(...), e.g.
        {
            "room": {"LOS": df1, "NLOS": df2},
            "office": {"LOS": df3, "NLOS": df4}
        }
    feature_col : str
        Column containing the 64 IQ-absolute values as a sequence per row.

    Returns
    -------
    dict
        {
            "X_room_LOS": np.ndarray,
            "X_room_NLOS": np.ndarray,
            "X_office_LOS": np.ndarray,
            "X_office_NLOS": np.ndarray
        }
    """
    out = {}

    for env in ["room", "office", "mixed", "proportional"]:
        for cls in ["LOS", "NLOS"]:
            df = split_dict[env][cls]

            # each row contains a sequence of length 64
            X = np.vstack(df[feature_col].to_numpy())

            out[f"X_{env}_{cls}"] = X

    return out

# arrays_3k = make_xy_arrays(split_3k, feature_col="iq_abs64")
# X_room_3k_LOS   = arrays_3k["X_room_LOS"]
# X_room_3k_NLOS  = arrays_3k["X_room_NLOS"]
# X_office_3k_LOS = arrays_3k["X_office_LOS"]
# X_office_3k_NLOS = arrays_3k["X_office_NLOS"]

# %%
def make_y_arrays(split_dict):
    """
    Create label arrays from split DataFrames.

    LOS  -> -1
    NLOS -> +1

    Parameters
    ----------
    split_dict : dict
        Output of split_los_nlos(...), e.g.
        {
            "room": {"LOS": df1, "NLOS": df2},
            "office": {"LOS": df3, "NLOS": df4}
        }

    Returns
    -------
    dict
        {
            "y_room_LOS": np.ndarray,
            "y_room_NLOS": np.ndarray,
            "y_office_LOS": np.ndarray,
            "y_office_NLOS": np.ndarray
        }
    """
    out = {}

    for env in ["room", "office", "mixed", "proportional"]:
        for cls in ["LOS", "NLOS"]:
            df = split_dict[env][cls]
            label = -1 if cls == "LOS" else +1
            y = np.full(len(df), label, dtype=int)
            out[f"y_{env}_{cls}"] = y

    return out

# %%
def make_train_test_data(split_dict, feature_col="iq_64abs", test_size=0.30, seed=2584):
    """
    Build train/test sets for room and office from split LOS/NLOS DataFrames.

    Parameters
    ----------
    split_dict : dict
        Output of split_los_nlos(...), e.g.
        {
            "room": {"LOS": df1, "NLOS": df2},
            "office": {"LOS": df3, "NLOS": df4}
        }
    feature_col : str
        Column containing the IQ-absolute sequence (length 64).
    test_size : float
        Fraction used for testing, e.g. 0.30.
    seed : int
        Random seed.

    Returns
    -------
    dict
        {
            "Xtr_room", "Xte_room", "Ytr_room", "Yte_room",
            "Xtr_office", "Xte_office", "Ytr_office", "Yte_office"
        }
    """
    out = {}
    for env in ["room", "office", "mixed", "proportional"]:
        df_los = split_dict[env]["LOS"]
        df_nlos = split_dict[env]["NLOS"]

        X_los = np.vstack(df_los[feature_col].to_numpy())
        X_nlos = np.vstack(df_nlos[feature_col].to_numpy())

        y_los = -np.ones(len(df_los), dtype=int)
        y_nlos = +np.ones(len(df_nlos), dtype=int)

        X = np.vstack([X_los, X_nlos])
        y = np.concatenate([y_los, y_nlos])

        Xtr, Xte, Ytr, Yte = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=y
        )

        # Save complete dataset
        out[f"X_{env}"] = X
        out[f"Y_{env}"] = y
        out[f"X_los_{env}"] = X_los
        out[f"Y_los_{env}"] = y_los
        out[f"X_nlos_{env}"] = X_nlos
        out[f"Y_nlos_{env}"] = y_nlos
        out[f"Xtr_{env}"] = Xtr
        out[f"Xte_{env}"] = Xte
        out[f"Ytr_{env}"] = Ytr
        out[f"Yte_{env}"] = Yte

    return out


