import numpy as np

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred))
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / np.where(denom == 0, 1e-8, denom))

def compute_metrics(y_true, y_pred):
    return {
        "mae": mae(y_true, y_pred),
        "smape": smape(y_true, y_pred)
    }