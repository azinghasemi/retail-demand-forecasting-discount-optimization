"""
Data preparation and cleaning functions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def clean_retail_data(df):
    """
    Clean retail sales data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw retail sales dataframe
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Convert date column to datetime
    if 'date' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['date'])
    elif 'Date' in df_clean.columns:
        df_clean['Date'] = pd.to_datetime(df_clean['Date'])
        df_clean.rename(columns={'Date': 'date'}, inplace=True)
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Remove outliers (using IQR method for sales/demand)
    target_cols = ['Demand', 'demand', 'sales', 'Sales', 'Unit Sold', 'Units Sold']
    target_col = None
    for col in target_cols:
        if col in df_clean.columns:
            target_col = col
            break
    
    if target_col:
        Q1 = df_clean[target_col].quantile(0.25)
        Q3 = df_clean[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[target_col] >= lower_bound) & 
                           (df_clean[target_col] <= upper_bound)]
        
        # Ensure target values are positive
        df_clean = df_clean[df_clean[target_col] > 0]
    
    print(f"Data cleaned: {df_clean.shape[0]} rows remaining")
    return df_clean


def validate_data(df):
    """
    Validate cleaned data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to validate
        
    Returns:
    --------
    bool
        True if data is valid
    """
    if df is None or df.empty:
        print("Error: Dataframe is empty")
        return False
    
    required_cols = ['date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        return False
    
    if df['date'].isnull().any():
        print("Warning: Missing dates found")
        return False
    
    print("Data validation passed")
    return True

