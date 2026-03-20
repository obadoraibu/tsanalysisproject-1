import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .utils import ensure_dir

def download_m4_monthly(data_dir: str, urls: Dict[str, str]):
    ensure_dir(data_dir)
    import requests
    
    for name, url in urls.items():
        filepath = os.path.join(data_dir, f"Monthly-{name}.csv")
        if not os.path.exists(filepath):
            response = requests.get(url)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(response.content)

def load_m4_monthly(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(data_dir, "Monthly-train.csv")
    test_path = os.path.join(data_dir, "Monthly-test.csv")
    info_path = os.path.join(data_dir, "Monthly-info.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    info_df = pd.read_csv(info_path)
    
    return train_df, test_df, info_df

def sample_series_ids(train_df: pd.DataFrame, sample_n: int, seed: int) -> List[str]:
    return train_df["V1"].sample(sample_n, random_state=seed).tolist()

def row_to_series(row: pd.Series) -> List[float]:
    values = row.drop("V1").dropna().astype(float).tolist()
    return values

def build_series_dicts(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                      sample_ids: List[str]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    train_sample = train_df[train_df["V1"].isin(sample_ids)].copy()
    test_sample = test_df[test_df["V1"].isin(sample_ids)].copy()
    
    series_dict = {
        row["V1"]: row_to_series(row) 
        for _, row in train_sample.iterrows()
    }
    
    test_dict = {
        row["V1"]: row_to_series(row) 
        for _, row in test_sample.iterrows()
    }
    
    return series_dict, test_dict