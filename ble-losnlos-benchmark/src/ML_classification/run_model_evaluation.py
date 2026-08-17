from src.ML_classification.build_model_specs import build_model_specs
from src.ML_classification.train_model import train_model
from src.ML_classification.predict_model import predict_model


def run_model_evaluation(
    datasets,
    best_params_all,
    seed=2584,
    class_order=(-1, +1),
):
    """
    Train and predict all model configurations
    for all datasets.

    Parameters
    ----------
    datasets : dict
        Example:
        {
            "room_3k": {
                "Xtr": ...,
                "Ytr": ...,
                "Xte": ...,
                "Yte": ...
            },
            ...
        }

    best_params_all : dict
        Best kernel parameters for each dataset.

    Returns
    -------
    trained_models_all : dict
        Trained models for every dataset.

    model_results_all : dict
        Predictions, confusion matrices and timing
        for every dataset and model.
    """

    trained_models_all = {}
    model_results_all = {}

    # =========================================================
    # Dataset loop
    # =========================================================
    for dataset_name, data in datasets.items():

        print("\n" + "#" * 70)
        print(f"DATASET: {dataset_name}")
        print("#" * 70)

        Xtr = data["Xtr"]
        Ytr = data["Ytr"]
        Xte = data["Xte"]
        Yte = data["Yte"]

        # -----------------------------------------------------
        # Get optimal kernel parameters for this dataset
        # -----------------------------------------------------
        best_params = best_params_all[dataset_name]

        # -----------------------------------------------------
        # Build model specifications
        # -----------------------------------------------------
        input_dim = Xtr.shape[1]

        ml_specs = build_model_specs(
            best_params=best_params,
            input_dim=input_dim,
        )

        trained_models = {}
        model_results = {}

        # =====================================================
        # Model loop
        # =====================================================
        for i, (name, spec) in enumerate(ml_specs, 1):

            # -------------------------
            # Train
            # -------------------------
            pipe, t_fit = train_model(
                spec=spec,
                Xtr=Xtr,
                Ytr=Ytr,
                seed=seed,
            )

            trained_models[name] = pipe

            # -------------------------
            # Predict
            # -------------------------
            yhat, cm, t_pred = predict_model(
                model=pipe,
                Xte=Xte,
                Yte=Yte,
                class_order=class_order,
            )

            # -------------------------
            # Store result
            # -------------------------
            model_results[name] = {
                "spec": spec,
                "yhat": yhat,
                "cm": cm,
                "fit_time_s": t_fit,
                "pred_time_s": t_pred,
            }

            # -------------------------
            # Print
            # -------------------------
            print(
                f"\n[{i:02d}/{len(ml_specs)}] "
                f"{dataset_name} | {name}"
            )

            print(
                f"Training time:   {t_fit:.3f} s"
            )

            print(
                f"Prediction time: {t_pred:.3f} s"
            )

            print("Confusion matrix:")
            print(cm)

        # -----------------------------------------------------
        # Save results for this dataset
        # -----------------------------------------------------
        trained_models_all[dataset_name] = trained_models
        model_results_all[dataset_name] = model_results

    return trained_models_all, model_results_all