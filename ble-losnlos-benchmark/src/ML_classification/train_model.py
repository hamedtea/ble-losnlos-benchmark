import time

from src.ML_classification.make_pipeline import make_pipeline


def train_model(
    spec,
    Xtr,
    Ytr,
    seed=2584,
):
    """
    Build and train one pipeline.

    Returns
    -------
    model :
        Trained sklearn pipeline.

    fit_time_s : float
        Training time in seconds.
    """

    input_dim = Xtr.shape[1]

    # Build pipeline
    pipe = make_pipeline(
        spec=spec,
        seed=seed,
        input_dim=input_dim,
    )

    # Train
    t0 = time.perf_counter()

    pipe.fit(
        Xtr,
        Ytr,
    )

    fit_time_s = time.perf_counter() - t0

    return pipe, fit_time_s

