import numpy as np
import pandas as pd


def make_mixed_proportional_dfs(
    df_room,
    df_office,
    seed=134,
):
    """
    Create dictionaries

        df_mixed["3k"]
        df_mixed["7k"]

        df_proportional["3k"]
        df_proportional["7k"]

    from Room and Office datasets.
    """

    rng = np.random.default_rng(seed)

    df_mixed = {}
    df_proportional = {}

    for size in ["3k", "7k"]:

        room = df_room[size].reset_index(drop=True)
        office = df_office[size].reset_index(drop=True)

        # --------------------------
        # MIXED
        # --------------------------
        pool = pd.concat(
            [room, office],
            ignore_index=True
        )

        idx = rng.permutation(len(pool))

        df_mixed[size] = (
            pool.iloc[idx]
            .reset_index(drop=True)
        )

        # --------------------------
        # PROPORTIONAL
        # (equal Room / Office)
        # --------------------------
        n_each = min(len(room), len(office))

        idx_room = rng.choice(
            len(room),
            size=n_each,
            replace=False
        )

        idx_office = rng.choice(
            len(office),
            size=n_each,
            replace=False
        )

        prop = pd.concat(
            [
                room.iloc[idx_room],
                office.iloc[idx_office],
            ],
            ignore_index=True,
        )

        perm = rng.permutation(len(prop))

        df_proportional[size] = (
            prop.iloc[perm]
            .reset_index(drop=True)
        )

    return df_mixed, df_proportional