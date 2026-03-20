import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer

def scale_series_dict(series_dict, scaler_type="standard"):
    scaled_dict = {}
    scalers = {}
    
    for ts_id, values in series_dict.items():
        arr = np.asarray(values, dtype=np.float32).reshape(-1, 1)
        
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        elif scaler_type == "quantile":
            n_quantiles = min(100, len(arr))
            scaler = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=n_quantiles,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        scaled = scaler.fit_transform(arr).flatten().astype(np.float32)
        scaled_dict[ts_id] = scaled
        scalers[ts_id] = scaler
    
    return scaled_dict, scalers

def inverse_transform_forecast(forecast, scaler):
    forecast = np.asarray(forecast).reshape(-1, 1)
    return scaler.inverse_transform(forecast).flatten()