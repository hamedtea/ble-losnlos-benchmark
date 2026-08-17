from src.data_io import read_room_office
from src.ml_dataset_generator import ml_dataset_generator
from src.preprocessing import split_los_nlos, make_xy_arrays, make_y_arrays, make_train_test_data
from src.explanatory.hexbin import plot_hexbin
from src.make_mixed_datasets import make_mixed_proportional_dfs

from src.scaler.scaler_check import report_scaling_checks, report_scaling_checks_all
from src.scaler.Xtrs import scale_training_data, add_scaled_training_data

from src.hyper_parameter_optimization.run_kernel_optimization import run_kernel_optimization

from src.hyper_parameter_optimization.kernel_specs import kernel_specs

from src.ML_classification.run_model_evaluation import run_model_evaluation


import matplotlib.pyplot as plt
import numpy as np

def main():
    # load raw data
    df_room, df_office = read_room_office()
    df_mixed, df_proportional = make_mixed_proportional_dfs(
    df_room,
    df_office,
    )
    # split into LOS / NLOS for 3k and 7k
    split_3k = split_los_nlos(df_room, df_office, df_mixed, df_proportional, "3k")
    split_7k = split_los_nlos(df_room, df_office, df_mixed, df_proportional, "7k")
    # feature arrays
    arrays_3k = make_xy_arrays(split_3k, feature_col="iq_64abs")
    arrays_7k = make_xy_arrays(split_7k, feature_col="iq_64abs")

    # label arrays
    labels_3k = make_y_arrays(split_3k)
    labels_7k = make_y_arrays(split_7k)

    # train/test data (70/30)
    data_3k = make_train_test_data(split_3k, feature_col="iq_64abs", test_size=0.30, seed=2584)
    data_7k = make_train_test_data(split_7k, feature_col="iq_64abs", test_size=0.30, seed=2584)


    # print shapes as a sanity check
    print("3K Room:", data_3k["X_room"].shape, data_3k["Y_room"].shape)
    print("3K Room: LOS", data_3k["X_los_room"].shape, data_3k["Y_los_room"].shape)
    print("3K Room: NLOS", data_3k["X_nlos_room"].shape, data_3k["Y_nlos_room"].shape)
    print("3K Room train:", data_3k["Xtr_room"].shape, data_3k["Ytr_room"].shape)
    print("3K Room test :", data_3k["Xte_room"].shape, data_3k["Yte_room"].shape)
    print("7K office:", data_3k["X_office"].shape, data_3k["Y_office"].shape)
    print("7K office: LOS", data_3k["X_los_office"].shape, data_3k["Y_los_office"].shape)
    print("3K office: NLOS", data_3k["X_nlos_office"].shape, data_3k["Y_nlos_office"].shape)
    print("3K Office train:", data_3k["Xtr_office"].shape, data_3k["Ytr_office"].shape)
    print("3K Office test :", data_3k["Xte_office"].shape, data_3k["Yte_office"].shape)

    print("7K Room:", data_7k["X_room"].shape, data_7k["Y_room"].shape)
    print("7K Room: LOS", data_7k["X_los_room"].shape, data_7k["Y_los_room"].shape)
    print("3K Room: NLOS", data_3k["X_nlos_room"].shape, data_3k["Y_nlos_room"].shape)
    print("7K Room train:", data_7k["Xtr_room"].shape, data_7k["Ytr_room"].shape)
    print("7K Room test :", data_7k["Xte_room"].shape, data_7k["Yte_room"].shape)
    print("7K office:", data_7k["X_office"].shape, data_7k["Y_office"].shape)
    print("7K office: LOS", data_7k["X_los_office"].shape, data_7k["Y_los_office"].shape)
    print("7K office: NLOS", data_7k["X_nlos_office"].shape, data_7k["Y_nlos_office"].shape)
    print("7K Office train:", data_7k["Xtr_office"].shape, data_7k["Ytr_office"].shape)
    print("7K Office test :", data_7k["Xte_office"].shape, data_7k["Yte_office"].shape)

    scaling_results = report_scaling_checks_all(
    data_3k=data_3k,
    data_7k=data_7k,
    robust_quantiles=(5, 95),
    outlier_thresh=1.28,
    )

    scaled_data = add_scaled_training_data(
    data_3k,
    data_7k,
    robust_quantiles=(5, 95),
    )

    kernel_datasets = {
    "room_3k": data_3k["Xtrs_room"],
    "office_3k": data_3k["Xtrs_office"],
    "room_7k": data_7k["Xtrs_room"],
    "office_7k": data_7k["Xtrs_office"],
    "mixed_3k": data_3k["Xtrs_mixed"],
    "mixed_7k": data_7k["Xtrs_mixed"],
    "proportional_3k": data_3k["Xtrs_proportional"],
    "proportional_7k": data_7k["Xtrs_proportional"],
    }

    # -----------------------------------------
    # Kernel optimization
    # -----------------------------------------
    kernel_results, best_params_all = run_kernel_optimization(
        datasets=kernel_datasets,
        kernel_specs=kernel_specs,
        components_grid=[18, 32, 64],
        sample_size=1000,
        random_state=432,
    )

    # --------------------------------------------------
    # Generate ML datasets
    # --------------------------------------------------
    ml_datasets = ml_dataset_generator(
        data_3k=data_3k,
        data_7k=data_7k,
    )

    print(ml_datasets.keys())

    trained_models_all, model_results_all = run_model_evaluation(
    datasets=ml_datasets,
    best_params_all=best_params_all,
    seed=2584,
    )


    for size_tag, data in [("3K", data_3k), ("7K", data_7k)]:

        plt.figure()
        plot_hexbin(
            data["X_los_room"],
            data["X_nlos_room"],
            title=f"Room {size_tag}: LOS vs NLOS"
        )
        
        plt.figure()
        plot_hexbin(
            data["X_los_office"],
            data["X_nlos_office"],
            title=f"Office {size_tag}: LOS vs NLOS"
        )
        plt.show()
    
    # return everything if needed
    return {
        "split_3k": split_3k,
        "split_7k": split_7k,
        "arrays_3k": arrays_3k,
        "arrays_7k": arrays_7k,
        "labels_3k": labels_3k,
        "labels_7k": labels_7k,
        "data_3k": data_3k,
        "data_7k": data_7k,
    }, kernel_results, best_params_all, trained_models
    


if __name__ == "__main__":
    result = main()