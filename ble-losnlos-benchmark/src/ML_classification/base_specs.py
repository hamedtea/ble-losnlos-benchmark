    base_specs = [
        ("Raw input",    dict(kernel="raw_input")),
        ("Scaler only", dict(kernel="scaler_only")),
        ("Linear",      dict(kernel="linear", n_pca=pca_dim)),

        ("RBF", dict(kernel="rbf",
                     gamma=float(best_params["rbf"]["gamma"]),
                     n_components=int(best_params["rbf"]["n_components"]))),

        ("Sigmoid", dict(kernel="sigmoid",
                         gamma=float(best_params["sigmoid"]["gamma"]),
                         coef0=0.0,
                         n_components=int(best_params["sigmoid"]["n_components"]))),

        ("Poly p=3", dict(kernel="polynomial",
                          gamma=float(best_params["polynomial"]["gamma"]),
                          degree=3, coef0=1.0,
                          n_components=int(best_params["polynomial"]["n_components"]))),

        ("Cosine", dict(kernel="cosine",
                        gamma=None,
                        n_components=int(best_params["cosine"]["n_components"]))),

        ("RFF", dict(kernel="rff",
                     gamma=float(best_params["rbf"]["gamma"]),
                     n_components=int(best_params["rbf"]["n_components"]))),
    ]

    # --- expand: same feature map with 3 downstream classifiers ---
    clf_variants = [
        ("SVC", dict(clf="svc")),
        ("RF",  dict(clf="rf", n_estimators=100, max_depth=5)),
        ("MLP", dict(clf="mlp", hidden_layer_sizes=(16, 8), max_iter=2000)),
    ]