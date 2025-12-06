"""
Utility functions for retail demand forecasting project.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
    """
    Load data from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None


def save_data(df, file_path, index=False):
    """
    Save dataframe to CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to save
    file_path : str
        Path to save the CSV file
    index : bool
        Whether to save index
    """
    try:
        df.to_csv(file_path, index=index)
        print(f"Data saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving data: {str(e)}")


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary with metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }
    
    return metrics


def print_metrics(metrics, model_name="Model"):
    """
    Print metrics in a formatted way.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with metrics
    model_name : str
        Name of the model
    """
    print(f"\n{model_name} Performance Metrics:")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("-" * 40)

