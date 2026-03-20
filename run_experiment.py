#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from config import *
from src.utils import set_seed, ensure_dir, save_dataframe, print_experiment_info
from src.data import download_m4_monthly, load_m4_monthly, sample_series_ids, build_series_dicts
from src.baselines import evaluate_baselines
from src.scaling import scale_series_dict
from src.catboost_model import evaluate_catboost_model
from src.lstm_model import evaluate_lstm_model
from src.evaluation import build_summary_table, analyze_scaling_impact
from src.plots import create_all_plots


def main():
    set_seed(SEED)
    ensure_dir(DATA_DIR)
    ensure_dir(RESULTS_DIR)
    ensure_dir(FIGURES_DIR)
    print_experiment_info(SAMPLE_N, INPUT_SIZE, HORIZON, SEED)
    
    download_m4_monthly(DATA_DIR, M4_URLS)
    train_df, test_df, info_df = load_m4_monthly(DATA_DIR)
    
    sample_ids = sample_series_ids(train_df, SAMPLE_N, SEED)
    series_dict, test_dict = build_series_dicts(train_df, test_df, sample_ids)
    
    baseline_results = evaluate_baselines(series_dict, test_dict, sample_ids, SEASON_LENGTH)
    save_dataframe(baseline_results, os.path.join(RESULTS_DIR, "baseline_results.csv"))
    
    catboost_results = []
    
    cb_none_results = evaluate_catboost_model(
        series_dict, test_dict, sample_ids, CATBOOST_PARAMS,
        model_name="catboost_none", input_size=INPUT_SIZE
    )
    catboost_results.append(cb_none_results)
    
    for scaler_type in ["standard", "robust", "quantile"]:
        scaled_series, scalers = scale_series_dict(series_dict, scaler_type)
        
        cb_scaled_results = evaluate_catboost_model(
            scaled_series, test_dict, sample_ids, CATBOOST_PARAMS,
            model_name=f"catboost_{scaler_type}", input_size=INPUT_SIZE,
            scalers=scalers
        )
        catboost_results.append(cb_scaled_results)
    
    catboost_all_results = pd.concat(catboost_results, ignore_index=True)
    save_dataframe(catboost_all_results, os.path.join(RESULTS_DIR, "catboost_results.csv"))
    
    lstm_results = []
    
    lstm_none_results = evaluate_lstm_model(
        series_dict, test_dict, sample_ids, LSTM_PARAMS,
        model_name="lstm_none", input_size=INPUT_SIZE
    )
    lstm_results.append(lstm_none_results)
    
    for scaler_type in ["standard", "robust", "quantile"]:
        scaled_series, scalers = scale_series_dict(series_dict, scaler_type)
        
        lstm_scaled_results = evaluate_lstm_model(
            scaled_series, test_dict, sample_ids, LSTM_PARAMS,
            model_name=f"lstm_{scaler_type}", input_size=INPUT_SIZE,
            scalers=scalers
        )
        lstm_results.append(lstm_scaled_results)
    
    lstm_all_results = pd.concat(lstm_results, ignore_index=True)
    save_dataframe(lstm_all_results, os.path.join(RESULTS_DIR, "lstm_results.csv"))
    
    final_results = pd.concat([
        baseline_results,
        catboost_all_results,
        lstm_all_results
    ], ignore_index=True)
    
    summary_table = build_summary_table(final_results)
    print(summary_table.to_string(index=False, float_format='%.3f'))
    
    save_dataframe(final_results, os.path.join(RESULTS_DIR, "final_results_per_series.csv"))
    save_dataframe(summary_table, os.path.join(RESULTS_DIR, "final_results_summary.csv"))
    
    scaling_impact = analyze_scaling_impact(final_results)
    save_dataframe(scaling_impact, os.path.join(RESULTS_DIR, "scaling_impact_analysis.csv"))
    
    create_all_plots(final_results, summary_table, FIGURES_DIR)
    
    lstm_none_smape = summary_table[summary_table['model'] == 'lstm_none']['smape'].iloc[0]
    lstm_scaled_smapes = summary_table[summary_table['model'].str.startswith('lstm_') & 
                                     (summary_table['model'] != 'lstm_none')]['smape']
    
    catboost_none_smape = summary_table[summary_table['model'] == 'catboost_none']['smape'].iloc[0]
    catboost_scaled_smapes = summary_table[summary_table['model'].str.startswith('catboost_') & 
                                          (summary_table['model'] != 'catboost_none')]['smape']
    
    lstm_improvement = lstm_none_smape - lstm_scaled_smapes.min()
    catboost_improvement = catboost_none_smape - catboost_scaled_smapes.min()
    
    print(f"LSTM improvement: {lstm_improvement:.3f}")
    print(f"CatBoost improvement: {catboost_improvement:.3f}")
    



if __name__ == "__main__":
    main()