import numpy as np
import pandas as pd
from typing import Dict, List
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
from .metrics import compute_metrics

def naive_forecast(train_series: List[float], horizon: int) -> np.ndarray:
    return np.repeat(train_series[-1], horizon)

def seasonal_naive_forecast(train_series: List[float], horizon: int, season_length: int = 12) -> np.ndarray:
    last_season = train_series[-season_length:]
    reps = int(np.ceil(horizon / season_length))
    forecast = np.tile(last_season, reps)[:horizon]
    return forecast

def auto_theta_forecast(train_series: List[float], horizon: int) -> np.ndarray:
    try:
        ts = pd.Series(train_series)
        theta_model = ThetaModel(ts, period=12)
        theta_fit = theta_model.fit()
        forecast = theta_fit.forecast(horizon)
        return np.array(forecast)
    except Exception:
        return naive_forecast(train_series, horizon)

def auto_ets_forecast(train_series: List[float], horizon: int, season_length: int = 12) -> np.ndarray:
    ts = pd.Series(train_series)
    best_fit = None
    best_aic = np.inf
    
    configs = [
        {"trend": None, "seasonal": None, "damped_trend": False},
        {"trend": "add", "seasonal": None, "damped_trend": False},
        {"trend": "add", "seasonal": None, "damped_trend": True},
        {"trend": None, "seasonal": "add", "damped_trend": False},
        {"trend": "add", "seasonal": "add", "damped_trend": False},
        {"trend": "add", "seasonal": "add", "damped_trend": True},
    ]
    
    for cfg in configs:
        try:
            seasonal_periods = season_length if cfg["seasonal"] is not None else None
            
            model = ExponentialSmoothing(
                ts,
                trend=cfg["trend"],
                seasonal=cfg["seasonal"],
                damped_trend=cfg["damped_trend"],
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
            
            fit = model.fit(optimized=True, use_brute=True)
            
            if np.isfinite(fit.aic) and fit.aic < best_aic:
                best_aic = fit.aic
                best_fit = fit
                
        except Exception:
            continue
    
    if best_fit is None:
        return naive_forecast(train_series, horizon)
    
    try:
        forecast = best_fit.forecast(horizon)
        return np.asarray(forecast, dtype=float)
    except Exception:
        return naive_forecast(train_series, horizon)

def evaluate_baselines(series_dict: Dict[str, List[float]], 
                      test_dict: Dict[str, List[float]], 
                      sample_ids: List[str],
                      season_length: int = 12) -> pd.DataFrame:
    results = []
    
    baseline_methods = {
        "naive": naive_forecast,
        "seasonal_naive": lambda ts, h: seasonal_naive_forecast(ts, h, season_length),
        "auto_theta": auto_theta_forecast,
        "auto_ets": lambda ts, h: auto_ets_forecast(ts, h, season_length),
    }
    
    for ts_id in sample_ids:
        train_series = series_dict[ts_id]
        test_series = test_dict[ts_id]
        horizon = len(test_series)
        
        for model_name, forecast_func in baseline_methods.items():
            try:
                forecast = forecast_func(train_series, horizon)
                metrics = compute_metrics(np.array(test_series), forecast)
                
                results.append({
                    "series_id": ts_id,
                    "model": model_name,
                    "mae": metrics["mae"],
                    "smape": metrics["smape"]
                })
                
            except Exception:
                forecast = naive_forecast(train_series, horizon)
                metrics = compute_metrics(np.array(test_series), forecast)
                
                results.append({
                    "series_id": ts_id,
                    "model": model_name,
                    "mae": metrics["mae"],
                    "smape": metrics["smape"]
                })
    
    return pd.DataFrame(results)