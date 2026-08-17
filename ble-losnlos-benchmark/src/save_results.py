import os
import joblib


def save_ml_results(
    trained_models_all,
    model_results_all,
    out_dir="./models",
):
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(
        trained_models_all,
        os.path.join(out_dir, "trained_models_all.joblib"),
    )
    print(f"Saved trained models to: {out_dir}")


def save_experiment_results(
    kernel_results,
    best_params_all,
    summary_results_all,
    out_dir="./results/summary",
):
    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(
        out_dir,
        "experiment_results.joblib",
    )

    joblib.dump(
        {
            "kernel_results": kernel_results,
            "best_params_all": best_params_all,
            "summary_results_all": summary_results_all,
        },
        save_path,
    )

    print(f"Saved summary results to: {out_dir}")