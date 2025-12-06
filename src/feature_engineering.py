"""
Feature engineering functions for retail demand forecasting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def create_temporal_features(df, date_col='date'):
    """
    Create temporal features from date column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with date column
    date_col : str
        Name of date column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with temporal features added
    """
    df_feat = df.copy()
    df_feat[date_col] = pd.to_datetime(df_feat[date_col])
    
    # Extract temporal features
    df_feat['year'] = df_feat[date_col].dt.year
    df_feat['month'] = df_feat[date_col].dt.month
    df_feat['day'] = df_feat[date_col].dt.day
    df_feat['day_of_week'] = df_feat[date_col].dt.dayofweek
    df_feat['day_of_year'] = df_feat[date_col].dt.dayofyear
    df_feat['week_of_year'] = df_feat[date_col].dt.isocalendar().week
    df_feat['quarter'] = df_feat[date_col].dt.quarter
    
    # Cyclical encoding for periodic features
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    df_feat['day_of_week_sin'] = np.sin(2 * np.pi * df_feat['day_of_week'] / 7)
    df_feat['day_of_week_cos'] = np.cos(2 * np.pi * df_feat['day_of_week'] / 7)
    
    # Is weekend
    df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
    
    # Is month end/start
    df_feat['is_month_start'] = df_feat[date_col].dt.is_month_start.astype(int)
    df_feat['is_month_end'] = df_feat[date_col].dt.is_month_end.astype(int)
    
    return df_feat


def create_lag_features(df, target_col='sales', lags=[1, 7, 14, 30], date_col='date'):
    """
    Create lag features for time series.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with sales data
    target_col : str
        Name of target column
    lags : list
        List of lag periods
    date_col : str
        Name of date column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with lag features added
    """
    df_lag = df.copy()
    df_lag = df_lag.sort_values(date_col)
    
    for lag in lags:
        df_lag[f'{target_col}_lag_{lag}'] = df_lag[target_col].shift(lag)
    
    return df_lag


def create_rolling_features(df, target_col='sales', windows=[7, 14, 30], date_col='date'):
    """
    Create rolling window features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with sales data
    target_col : str
        Name of target column
    windows : list
        List of window sizes
    date_col : str
        Name of date column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with rolling features added
    """
    df_roll = df.copy()
    df_roll = df_roll.sort_values(date_col)
    
    for window in windows:
        df_roll[f'{target_col}_rolling_mean_{window}'] = df_roll[target_col].rolling(
            window=window, min_periods=1).mean()
        df_roll[f'{target_col}_rolling_std_{window}'] = df_roll[target_col].rolling(
            window=window, min_periods=1).std()
        df_roll[f'{target_col}_rolling_max_{window}'] = df_roll[target_col].rolling(
            window=window, min_periods=1).max()
        df_roll[f'{target_col}_rolling_min_{window}'] = df_roll[target_col].rolling(
            window=window, min_periods=1).min()
    
    return df_roll


def create_discount_features(df, discount_col='discount'):
    """
    Create discount-related features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with discount data
    discount_col : str
        Name of discount column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with discount features added
    """
    df_disc = df.copy()
    
    if discount_col in df_disc.columns:
        df_disc['has_discount'] = (df_disc[discount_col] > 0).astype(int)
        df_disc['discount_category'] = pd.cut(
            df_disc[discount_col],
            bins=[-np.inf, 0, 10, 20, 30, np.inf],
            labels=['No Discount', 'Low (0-10%)', 'Medium (10-20%)', 
                   'High (20-30%)', 'Very High (>30%)']
        )
    else:
        df_disc['has_discount'] = 0
        df_disc['discount_category'] = 'No Discount'
    
    return df_disc


def prepare_features(df, target_col='sales', date_col='date', 
                    include_lags=True, include_rolling=True):
    """
    Prepare all features for modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    date_col : str
        Name of date column
    include_lags : bool
        Whether to include lag features
    include_rolling : bool
        Whether to include rolling features
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with all features
    """
    df_feat = df.copy()
    
    # Create temporal features
    df_feat = create_temporal_features(df_feat, date_col)
    
    # Create discount features
    df_feat = create_discount_features(df_feat)
    
    # Create lag features
    if include_lags and target_col in df_feat.columns:
        df_feat = create_lag_features(df_feat, target_col, date_col=date_col)
    
    # Create rolling features
    if include_rolling and target_col in df_feat.columns:
        df_feat = create_rolling_features(df_feat, target_col, date_col=date_col)
    
    # Fill NaN values from lag and rolling features
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
    df_feat[numeric_cols] = df_feat[numeric_cols].fillna(0)
    
    return df_feat

