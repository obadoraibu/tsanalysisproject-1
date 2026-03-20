import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional
from .evaluation import split_model_name

def setup_plot_style():
    plt.rcParams['figure.figsize'] = (10, 6)

def plot_mean_smape(summary_table: pd.DataFrame, save_path: Optional[str] = None):
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    df_sorted = summary_table.sort_values('smape', ascending=True)
    ax.barh(range(len(df_sorted)), df_sorted['smape'])
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['model'])
    ax.set_xlabel('Mean sMAPE (%)')
    ax.set_title('Mean sMAPE by Model')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


def plot_per_series_boxplot(final_results: pd.DataFrame, save_path: Optional[str] = None):
    setup_plot_style()
    
    key_models = [
        'auto_ets', 'auto_theta', 'catboost_none', 'catboost_standard',
        'lstm_none', 'lstm_standard', 'lstm_robust'
    ]
    
    df_filtered = final_results[final_results['model'].isin(key_models)].copy()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.boxplot([
        df_filtered[df_filtered['model'] == model]['smape'].values 
        for model in key_models
    ], labels=key_models)
    
    ax.set_ylabel('sMAPE (%)')
    ax.set_title('sMAPE Distribution by Model')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def plot_scaling_comparison(final_results: pd.DataFrame, save_path: Optional[str] = None):
    setup_plot_style()
    
    df_split = split_model_name(final_results)
    
    models_with_scaling = ["catboost", "lstm"]
    df_filtered = df_split[df_split["base_model"].isin(models_with_scaling)].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, base_model in enumerate(models_with_scaling):
        model_data = df_filtered[df_filtered["base_model"] == base_model]
        scaling_means = model_data.groupby("scaling")["smape"].mean().reset_index()
        
        axes[i].bar(scaling_means["scaling"], scaling_means["smape"])
        axes[i].set_title(f'{base_model}')
        axes[i].set_ylabel('Mean sMAPE (%)')
        axes[i].set_xlabel('Scaling Method')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def create_all_plots(final_results: pd.DataFrame, summary_table: pd.DataFrame, 
                    figures_dir: str):
    import os
    from .utils import ensure_dir
    
    ensure_dir(figures_dir)
    
    plot_mean_smape(summary_table, 
                   os.path.join(figures_dir, "mean_smape_barplot.png"))
    
    plot_per_series_boxplot(final_results, 
                           os.path.join(figures_dir, "per_series_boxplot.png"))
    
    plot_scaling_comparison(final_results, 
                          os.path.join(figures_dir, "scaling_comparison.png"))