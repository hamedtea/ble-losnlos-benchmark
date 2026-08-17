from src.data_io import read_room_office
from src.ml_dataset_generator import ml_dataset_generator
from src.preprocessing import split_los_nlos, make_xy_arrays, make_y_arrays, make_train_test_data
from src.make_mixed_datasets import make_mixed_proportional_dfs

from src.explanatory.learning_curve import learning_curve_comparison

from src.scaler.scaler_check import report_scaling_checks, report_scaling_checks_all
from src.scaler.Xtrs import scale_training_data, add_scaled_training_data

from src.hyper_parameter_optimization.kernel_specs import kernel_specs

from src.hyper_parameter_optimization.run_kernel_optimization import run_kernel_optimization

from src.ML_classification.run_model_evaluation import run_model_evaluation
from src.ML_classification.summary_report import summary_report
from src.ML_classification.mean_std_summary import mean_std_summary

from src.plotters.plot_hexbin_all_datasets import plot_hexbin_all_datasets
from src.plotters.learning_curve_visualizer import learning_curve_visualizer



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


    # --------------------------------------------------
    # Learning curve: ROOM and OFFICE
    # --------------------------------------------------
    lc_room = learning_curve_comparison(
        data_3k=data_3k,
        data_7k=data_7k,
        dataset_type="room",
        min_size=500,
        step=500,
        cv=5,
        scoring="accuracy",
    )

    lc_office = learning_curve_comparison(
        data_3k=data_3k,
        data_7k=data_7k,
        dataset_type="office",
        min_size=500,
        step=500,
        cv=5,
        scoring="accuracy",
    )

    # -----------------------------------------
    # Scaler check-in
    # -----------------------------------------
    scaling_results = report_scaling_checks_all(
    data_3k=data_3k,
    data_7k=data_7k,
    robust_quantiles=(5, 95),
    outlier_thresh=1.28,
    )

    # -----------------------------------------
    # Scaler check-in
    # -----------------------------------------
    scaled_data = add_scaled_training_data(
    data_3k,
    data_7k,
    robust_quantiles=(5, 95),
    )

    # -----------------------------------------
    # Kernel dataset generation
    # -----------------------------------------
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

    # --------------------------------------------------
    # ML pipeline training, prediction, evaluation
    # --------------------------------------------------
    trained_models_all, model_results_all = run_model_evaluation(
    datasets=ml_datasets,
    best_params_all=best_params_all,
    seed=2584,
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    summary_results_all = summary_report(
        ml_datasets=ml_datasets,
        model_results_all=model_results_all,
        trained_models_all=trained_models_all,
    )
    df_combined = mean_std_summary(
    summary_results_all
    )
    # Model becomes normal column
    df_combined = df_combined.reset_index()
    print("\nMean ± STD across datasets:")
    print(df_combined)

    # --------------------------------------------------
    # PLOTTERS
    # --------------------------------------------------

    # --------------------------------------------------
    # Plotting Hexabin graph for 3K and 7K
    # --------------------------------------------------
    figures_3k = plot_hexbin_all_datasets(
    data=data_3k
    )
    figures_7k = plot_hexbin_all_datasets(
    data=data_7k
    )
    print("Reached plotting section")
    print("Open figures:", plt.get_fignums())




    learning_results = {
    "room": lc_room,
    "office": lc_office,    
    }
    learning_curve_visualizer(
        learning_results=learning_results,
        title="Learning Curves: 3K to 7K Dataset Size",
    )


    for num in plt.get_fignums():
        fig = plt.figure(num)
        print(f"Figure {num}: {fig}")
    
    plt.show(block=True)


    return {
        "split_3k": split_3k,
        "split_7k": split_7k,
        "arrays_3k": arrays_3k,
        "arrays_7k": arrays_7k,
        "labels_3k": labels_3k,
        "labels_7k": labels_7k,
        "data_3k": data_3k,
        "data_7k": data_7k,
    }, kernel_results, best_params_all, trained_models_all, model_results_all,  summary_results_all 
    


if __name__ == "__main__":
    result = main()