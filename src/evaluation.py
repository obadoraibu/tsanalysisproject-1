import pandas as pd
import numpy as np
from typing import List

def build_summary_table(final_results: pd.DataFrame) -> pd.DataFrame:
    summary_table = (
        final_results
        .groupby("model")[["mae", "smape"]]
        .mean()
        .sort_values("smape")
        .reset_index()
    )
    
    return summary_table

def split_model_name(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    base_model_mapping = {
        "naive": ("baseline", "none"),
        "seasonal_naive": ("baseline_seasonal", "none"),
        "auto_theta": ("baseline_theta", "none"),
        "auto_ets": ("baseline_ets", "none"),
        "catboost_none": ("catboost", "none"),
        "catboost_standard": ("catboost", "standard"),
        "catboost_robust": ("catboost", "robust"),
        "catboost_quantile": ("catboost", "quantile"),
        "lstm_none": ("lstm", "none"),
        "lstm_standard": ("lstm", "standard"),
        "lstm_robust": ("lstm", "robust"),
        "lstm_quantile": ("lstm", "quantile"),
    }
    
    df[["base_model", "scaling"]] = df["model"].map(
        lambda x: base_model_mapping.get(x, (x, "none"))
    ).apply(pd.Series)
    
    scaling_order = ["none", "standard", "robust", "quantile"]
    df["scaling"] = pd.Categorical(df["scaling"], categories=scaling_order, ordered=True)
    
    return df

def compute_delta_vs_none(summary_table: pd.DataFrame) -> pd.DataFrame:
    df_split = split_model_name(summary_table)
    
    models_with_scaling = ["catboost", "lstm"]
    df_filtered = df_split[df_split["base_model"].isin(models_with_scaling)].copy()
    
    baseline_values = (
        df_filtered[df_filtered["scaling"] == "none"]
        .set_index("base_model")[["mae", "smape"]]
        .rename(columns={"mae": "mae_none", "smape": "smape_none"})
    )
    
    df_merged = df_filtered.merge(baseline_values, on="base_model", how="left")
    
    df_merged["delta_mae"] = df_merged["mae"] - df_merged["mae_none"]
    df_merged["delta_smape"] = df_merged["smape"] - df_merged["smape_none"]
    
    delta_df = df_merged[df_merged["scaling"] != "none"].copy()
    
    return delta_df[["model", "base_model", "scaling", "delta_mae", "delta_smape"]]

def pairwise_win_rate(final_results: pd.DataFrame, 
                     model_a: str, 
                     model_b: str, 
                     metric: str = "smape") -> float:
    results_a = final_results[final_results["model"] == model_a].set_index("series_id")[metric]
    results_b = final_results[final_results["model"] == model_b].set_index("series_id")[metric]
    
    common_series = results_a.index.intersection(results_b.index)
    
    if len(common_series) == 0:
        return np.nan
    
    wins = (results_a.loc[common_series] < results_b.loc[common_series]).sum()
    
    return wins / len(common_series)

def get_best_scaling_per_model(final_results: pd.DataFrame) -> pd.DataFrame:
    df_split = split_model_name(final_results)
    
    grouped = (
        df_split.groupby(["base_model", "scaling"])["smape"]
        .mean()
        .reset_index()
    )
    
    best_scaling = (
        grouped.loc[grouped.groupby("base_model")["smape"].idxmin()]
        .reset_index(drop=True)
    )
    
    return best_scaling

def analyze_scaling_impact(final_results: pd.DataFrame) -> pd.DataFrame:
    df_split = split_model_name(final_results)
    
    models_with_scaling = ["catboost", "lstm"]
    df_filtered = df_split[df_split["base_model"].isin(models_with_scaling)].copy()
    
    scaling_impact = []
    
    for base_model in models_with_scaling:
        model_data = df_filtered[df_filtered["base_model"] == base_model]
        
        scaling_means = model_data.groupby("scaling")["smape"].mean()
        
        if "none" in scaling_means:
            baseline = scaling_means["none"]
            
            for scaling_method in ["standard", "robust", "quantile"]:
                if scaling_method in scaling_means:
                    improvement = baseline - scaling_means[scaling_method]
                    improvement_pct = (improvement / baseline) * 100
                    
                    scaling_impact.append({
                        "base_model": base_model,
                        "scaling": scaling_method,
                        "baseline_smape": baseline,
                        "scaled_smape": scaling_means[scaling_method],
                        "improvement": improvement,
                        "improvement_pct": improvement_pct
                    })
    
    return pd.DataFrame(scaling_impact)