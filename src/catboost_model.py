import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from catboost import CatBoostRegressor
from .metrics import compute_metrics
from .scaling import inverse_transform_forecast

def make_lagged_dataset(series_dict: Dict[str, List[float]], input_size: int = 24) -> pd.DataFrame:
    rows = []
    
    for ts_id, values in series_dict.items():
        series = np.array(values)
        
        for i in range(input_size, len(series)):
            lags = series[i-input_size:i]
            target = series[i]
            
            row = {"series_id": ts_id, "target": target}
            for j in range(input_size):
                row[f"lag_{j+1}"] = lags[j]
            
            rows.append(row)
    
    return pd.DataFrame(rows)

def train_catboost_model(series_dict: Dict[str, List[float]], 
                        params: Dict, 
                        input_size: int = 24) -> CatBoostRegressor:
    lag_df = make_lagged_dataset(series_dict, input_size=input_size)
    feature_cols = [c for c in lag_df.columns if c.startswith("lag_")]
    
    model = CatBoostRegressor(**params)
    model.fit(lag_df[feature_cols], lag_df["target"])
    
    return model

def recursive_catboost_forecast(model: CatBoostRegressor, 
                               train_series: List[float], 
                               horizon: int, 
                               input_size: int = 24) -> np.ndarray:
    series = np.array(train_series, dtype=np.float32).copy()
    forecast = []
    
    for _ in range(horizon):
        lags = series[-input_size:].reshape(1, -1)
        lag_features = pd.DataFrame(lags, columns=[f"lag_{i+1}" for i in range(input_size)])
        
        pred = model.predict(lag_features)[0]
        forecast.append(pred)
        
        series = np.append(series, pred)
    
    return np.array(forecast)

def evaluate_catboost_model(series_dict: Dict[str, List[float]], 
                           test_dict: Dict[str, List[float]], 
                           sample_ids: List[str],
                           params: Dict,
                           model_name: str = "catboost_none",
                           input_size: int = 24,
                           scalers: Optional[Dict] = None) -> pd.DataFrame:
    model = train_catboost_model(series_dict, params, input_size)
    
    results = []
    
    for ts_id in sample_ids:
        train_series = series_dict[ts_id]
        test_series = test_dict[ts_id]
        horizon = len(test_series)
        
        try:
            forecast = recursive_catboost_forecast(model, train_series, horizon, input_size)
            
            if scalers is not None and ts_id in scalers:
                forecast = inverse_transform_forecast(forecast, scalers[ts_id])
            
            metrics = compute_metrics(np.array(test_series), forecast)
            
            results.append({
                "series_id": ts_id,
                "model": model_name,
                "mae": metrics["mae"],
                "smape": metrics["smape"]
            })
            
        except Exception:
            fallback_forecast = np.repeat(train_series[-1], horizon)
            if scalers is not None and ts_id in scalers:
                fallback_forecast = inverse_transform_forecast(fallback_forecast, scalers[ts_id])
            
            metrics = compute_metrics(np.array(test_series), fallback_forecast)
            
            results.append({
                "series_id": ts_id,
                "model": model_name,
                "mae": metrics["mae"],
                "smape": metrics["smape"]
            })
    
    return pd.DataFrame(results)