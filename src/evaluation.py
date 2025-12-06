"""
Model evaluation functions.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def calculate_all_metrics(y_true, y_pred):
    """
    Calculate comprehensive regression metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary with all metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    # Mean Error (bias)
    mean_error = np.mean(y_pred - y_true)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'Mean Error': mean_error
    }
    
    return metrics


def print_evaluation_report(y_true, y_pred, model_name="Model"):
    """
    Print comprehensive evaluation report.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model
    """
    metrics = calculate_all_metrics(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation Report")
    print(f"{'='*50}")
    print(f"Mean Absolute Error (MAE):     {metrics['MAE']:.4f}")
    print(f"Mean Squared Error (MSE):      {metrics['MSE']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
    print(f"R² Score:                      {metrics['R2']:.4f}")
    print(f"Mean Absolute % Error (MAPE):  {metrics['MAPE']:.2f}%")
    print(f"Mean Error (Bias):             {metrics['Mean Error']:.4f}")
    print(f"{'='*50}\n")
    
    return metrics


def plot_predictions_vs_actual(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot predictions vs actual values.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model
    save_path : str
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Values', fontsize=12)
    axes[0].set_ylabel('Predicted Values', fontsize=12)
    axes[0].set_title(f'{model_name}: Predictions vs Actual', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_pred - y_true
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values', fontsize=12)
    axes[1].set_ylabel('Residuals', fontsize=12)
    axes[1].set_title(f'{model_name}: Residuals Plot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()
    return fig


def compare_models(models_results):
    """
    Compare multiple models and return comparison dataframe.
    
    Parameters:
    -----------
    models_results : dict
        Dictionary with model names as keys and metrics dict as values
        
    Returns:
    --------
    pd.DataFrame
        Comparison dataframe
    """
    comparison_data = []
    
    for model_name, metrics in models_results.items():
        comparison_data.append({
            'Model': model_name,
            'MAE': metrics.get('MAE', np.nan),
            'RMSE': metrics.get('RMSE', np.nan),
            'R2': metrics.get('R2', np.nan),
            'MAPE': metrics.get('MAPE', np.nan)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('RMSE')
    
    return comparison_df

