kernel_specs = [

    {
        "name": "Nyström RBF",
        "method": "nystroem",
        "kernel": "rbf",
    },

    {
        "name": "Nyström Sigmoid",
        "method": "nystroem",
        "kernel": "sigmoid",
        "coef0": 0.0,
    },

    {
        "name": "Nyström Poly-3",
        "method": "nystroem",
        "kernel": "polynomial",
        "degree": 3,
        "coef0": 1.0,
    },

    {
        "name": "Nyström Cosine",
        "method": "nystroem",
        "kernel": "cosine",
    },

    {
        "name": "RFF RBF",
        "method": "rff",
        "kernel": "rbf",
    },
]

