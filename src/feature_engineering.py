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


def create_interpolated_order(df, order_col='Units Ordered', date_col='date'):
    """
    Create Interpolated Order feature to handle sparse order data.
    
    Because "Units Ordered" had many zeros, we create:
    Interpolated_Order = Last_Order_Amount / Days_Since_Last_Order
    
    This helps reduce sparsity and improves model performance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with order data
    order_col : str
        Name of units ordered column
    date_col : str
        Name of date column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with interpolated order feature added
    """
    df_int = df.copy()
    df_int = df_int.sort_values(date_col)
    
    if order_col not in df_int.columns:
        df_int['Interpolated_Order'] = 0
        return df_int
    
    # Initialize the interpolated order column
    df_int['Interpolated_Order'] = 0.0
    
    # Group by product/store if available to calculate per product/store
    group_cols = []
    if 'Product ID' in df_int.columns:
        group_cols.append('Product ID')
    if 'Store ID' in df_int.columns:
        group_cols.append('Store ID')
    
    if group_cols:
        for name, group in df_int.groupby(group_cols):
            last_order_amount = 0
            days_since_last_order = 1
            
            for idx, row in group.iterrows():
                if row[order_col] > 0:
                    last_order_amount = row[order_col]
                    days_since_last_order = 1
                    df_int.loc[idx, 'Interpolated_Order'] = last_order_amount
                else:
                    if days_since_last_order > 0 and last_order_amount > 0:
                        df_int.loc[idx, 'Interpolated_Order'] = last_order_amount / days_since_last_order
                    days_since_last_order += 1
    else:
        # If no grouping columns, calculate globally
        last_order_amount = 0
        days_since_last_order = 1
        
        for idx, row in df_int.iterrows():
            if row[order_col] > 0:
                last_order_amount = row[order_col]
                days_since_last_order = 1
                df_int.loc[idx, 'Interpolated_Order'] = last_order_amount
            else:
                if days_since_last_order > 0 and last_order_amount > 0:
                    df_int.loc[idx, 'Interpolated_Order'] = last_order_amount / days_since_last_order
                days_since_last_order += 1
    
    return df_int


def create_season_features(df, date_col='date'):
    """
    Create season features from date.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with date column
    date_col : str
        Name of date column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with season features added
    """
    df_season = df.copy()
    df_season[date_col] = pd.to_datetime(df_season[date_col])
    
    # Map months to seasons
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    
    df_season['Season'] = df_season[date_col].dt.month.map(season_map)
    
    return df_season


def create_holiday_features(df, date_col='date'):
    """
    Create holiday features using US Federal Holiday calendar.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with date column
    date_col : str
        Name of date column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with holiday features added
    """
    try:
        from pandas.tseries.holiday import USFederalHolidayCalendar
        from pandas.tseries.offsets import CustomBusinessDay
        
        df_hol = df.copy()
        df_hol[date_col] = pd.to_datetime(df_hol[date_col])
        
        # Create US Federal Holiday calendar
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=df_hol[date_col].min(), end=df_hol[date_col].max())
        
        # Mark holidays
        df_hol['is_holiday'] = df_hol[date_col].isin(holidays).astype(int)
        
        return df_hol
    except ImportError:
        print("Warning: Could not import holiday calendar. Setting is_holiday to 0.")
        df_hol = df.copy()
        df_hol['is_holiday'] = 0
        return df_hol


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


def prepare_features(df, target_col='Demand', date_col='date', 
                    include_lags=True, include_rolling=True, 
                    include_interpolated_order=True):
    """
    Prepare all features for modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column (default: 'Demand' as per report)
    date_col : str
        Name of date column
    include_lags : bool
        Whether to include lag features
    include_rolling : bool
        Whether to include rolling features
    include_interpolated_order : bool
        Whether to include interpolated order feature
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with all features
    """
    df_feat = df.copy()
    
    # Create temporal features
    df_feat = create_temporal_features(df_feat, date_col)
    
    # Create season features
    df_feat = create_season_features(df_feat, date_col)
    
    # Create holiday features
    df_feat = create_holiday_features(df_feat, date_col)
    
    # Create interpolated order feature (custom feature from report)
    if include_interpolated_order:
        df_feat = create_interpolated_order(df_feat, date_col=date_col)
    
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

