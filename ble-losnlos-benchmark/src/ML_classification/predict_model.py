import time
from sklearn.metrics import confusion_matrix


def predict_model(
    model,
    Xte,
    Yte,
    class_order=(-1, +1),
):
    """
    Predict test labels and calculate confusion matrix.

    Returns
    -------
    yhat :
        Predicted labels.

    cm :
        Confusion matrix.

    pred_time_s : float
        Prediction time in seconds.
    """

    # Predict
    t0 = time.perf_counter()

    yhat = model.predict(Xte)

    pred_time_s = time.perf_counter() - t0

    # Confusion matrix
    cm = confusion_matrix(
        Yte,
        yhat,
        labels=list(class_order),
    )

    return yhat, cm, pred_time_s