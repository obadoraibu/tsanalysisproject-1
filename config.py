import os

SEED = 42
SAMPLE_N = 300
INPUT_SIZE = 24
HORIZON = 18
SEASON_LENGTH = 12

CATBOOST_PARAMS = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "MAE",
    "verbose": 0,
    "random_seed": SEED,
}

LSTM_PARAMS = {
    "hidden_dim": 8,
    "num_layers": 1,
    "dropout": 0.0,
    "epochs": 3,
    "batch_size": 16,
    "lr": 1e-3,
}

SCALERS = ["none", "standard", "robust", "quantile"]
DATA_DIR = "data"
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

M4_URLS = {
    "train": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Monthly-train.csv",
    "test": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/Monthly-test.csv",
    "info": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/M4-info.csv"
}