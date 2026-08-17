
def ml_dataset_generator(data_3k, data_7k):
    """
    Build ML train/test dataset dictionaries for:
    - room
    - office
    - proportional
    - mixed

    for both 3k and 7k datasets.
    """

    ml_datasets = {
        # --------------------------------------------------
        # Room
        # --------------------------------------------------
        "room_3k": {
            "Xtr": data_3k["Xtr_room"],
            "Ytr": data_3k["Ytr_room"],
            "Xte": data_3k["Xte_room"],
            "Yte": data_3k["Yte_room"],
        },

        "room_7k": {
            "Xtr": data_7k["Xtr_room"],
            "Ytr": data_7k["Ytr_room"],
            "Xte": data_7k["Xte_room"],
            "Yte": data_7k["Yte_room"],
        },

        # --------------------------------------------------
        # Office
        # --------------------------------------------------
        "office_3k": {
            "Xtr": data_3k["Xtr_office"],
            "Ytr": data_3k["Ytr_office"],
            "Xte": data_3k["Xte_office"],
            "Yte": data_3k["Yte_office"],
        },

        "office_7k": {
            "Xtr": data_7k["Xtr_office"],
            "Ytr": data_7k["Ytr_office"],
            "Xte": data_7k["Xte_office"],
            "Yte": data_7k["Yte_office"],
        },

        # --------------------------------------------------
        # Proportional
        # --------------------------------------------------
        "proportional_3k": {
            "Xtr": data_3k["Xtr_proportional"],
            "Ytr": data_3k["Ytr_proportional"],
            "Xte": data_3k["Xte_proportional"],
            "Yte": data_3k["Yte_proportional"],
        },

        "proportional_7k": {
            "Xtr": data_7k["Xtr_proportional"],
            "Ytr": data_7k["Ytr_proportional"],
            "Xte": data_7k["Xte_proportional"],
            "Yte": data_7k["Yte_proportional"],
        },

        # --------------------------------------------------
        # Mixed
        # --------------------------------------------------
        "mixed_3k": {
            "Xtr": data_3k["Xtr_mixed"],
            "Ytr": data_3k["Ytr_mixed"],
            "Xte": data_3k["Xte_mixed"],
            "Yte": data_3k["Yte_mixed"],
        },

        "mixed_7k": {
            "Xtr": data_7k["Xtr_mixed"],
            "Ytr": data_7k["Ytr_mixed"],
            "Xte": data_7k["Xte_mixed"],
            "Yte": data_7k["Yte_mixed"],
        },
    }

    return ml_datasets