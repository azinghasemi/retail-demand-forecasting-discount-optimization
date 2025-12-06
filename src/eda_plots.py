"""
Exploratory Data Analysis plotting functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_sales_over_time(df, date_col='date', sales_col='sales', save_path=None):
    """
    Plot sales over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with sales data
    date_col : str
        Name of date column
    sales_col : str
        Name of sales column
    save_path : str
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    df_sorted = df.sort_values(date_col)
    ax.plot(df_sorted[date_col], df_sorted[sales_col], linewidth=1.5)
    ax.set_title('Sales Over Time', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sales', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()
    return fig


def plot_seasonal_heatmap(df, date_col='date', sales_col='sales', save_path=None):
    """
    Create seasonal heatmap of sales.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with sales data
    date_col : str
        Name of date column
    sales_col : str
        Name of sales column
    save_path : str
        Path to save the plot
    """
    df_plot = df.copy()
    df_plot['year'] = pd.to_datetime(df_plot[date_col]).dt.year
    df_plot['month'] = pd.to_datetime(df_plot[date_col]).dt.month
    df_plot['day_of_week'] = pd.to_datetime(df_plot[date_col]).dt.dayofweek
    
    # Create pivot table for heatmap
    pivot = df_plot.pivot_table(values=sales_col, index='month', 
                                columns='day_of_week', aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', 
                cbar_kws={'label': 'Average Sales'}, ax=ax)
    ax.set_title('Seasonal Sales Heatmap (Month vs Day of Week)', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Day of Week (0=Monday, 6=Sunday)', fontsize=12)
    ax.set_ylabel('Month', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()
    return fig


def plot_correlation_heatmap(df, save_path=None):
    """
    Plot correlation heatmap of numeric features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with numeric features
    save_path : str
        Path to save the plot
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        print("No numeric columns found for correlation")
        return
    
    corr_matrix = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, 
                cbar_kws={'label': 'Correlation Coefficient'}, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()
    return fig


def plot_feature_importance(importance_dict, save_path=None):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    importance_dict : dict
        Dictionary with feature names and importance scores
    save_path : str
        Path to save the plot
    """
    features = list(importance_dict.keys())
    importances = list(importance_dict.values())
    
    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]
    features = [features[i] for i in sorted_idx]
    importances = [importances[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(features)), importances)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance', fontsize=16, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()
    return fig


def plot_model_comparison(models_metrics, save_path=None):
    """
    Plot comparison of different models.
    
    Parameters:
    -----------
    models_metrics : dict
        Dictionary with model names and their metrics
    save_path : str
        Path to save the plot
    """
    models = list(models_metrics.keys())
    metrics = list(models_metrics.values())
    
    # Extract RMSE for comparison
    rmse_values = [m.get('RMSE', 0) for m in metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, rmse_values, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Model Comparison (RMSE)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, max(rmse_values) * 1.2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()
    return fig

