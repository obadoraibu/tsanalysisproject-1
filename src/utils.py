import os
import random
import numpy as np
import torch
import pandas as pd

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_dataframe(df: pd.DataFrame, filepath: str):
    ensure_dir(os.path.dirname(filepath))
    df.to_csv(filepath, index=False, float_format='%.6f')

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_experiment_info(sample_n: int, input_size: int, horizon: int, seed: int):
    device = get_device()
    print(f"Device: {device}, Sample: {sample_n}, Input: {input_size}, Horizon: {horizon}, Seed: {seed}")