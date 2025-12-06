"""
Linear Regression model for demand forecasting.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def train_linear_regression(X, y, test_size=0.2, random_state=42):
    """
    Train linear regression model.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.array
        Feature matrix
    y : pd.Series or np.array
        Target variable
    test_size : float
        Proportion of test set
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    tuple
        (model, X_train, X_test, y_train, y_test, scaler)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    print("Linear Regression model trained successfully")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return model, X_train, X_test, y_train, y_test, scaler


def predict_linear_regression(model, X, scaler):
    """
    Make predictions using linear regression model.
    
    Parameters:
    -----------
    model : LinearRegression
        Trained model
    X : pd.DataFrame or np.array
        Feature matrix
    scaler : StandardScaler
        Fitted scaler
        
    Returns:
    --------
    np.array
        Predictions
    """
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    return predictions


def get_feature_importance(model, feature_names):
    """
    Get feature importance (coefficients) from linear regression.
    
    Parameters:
    -----------
    model : LinearRegression
        Trained model
    feature_names : list
        List of feature names
        
    Returns:
    --------
    dict
        Dictionary with feature names and coefficients
    """
    importance = dict(zip(feature_names, model.coef_))
    # Sort by absolute value
    importance = dict(sorted(importance.items(), 
                            key=lambda x: abs(x[1]), reverse=True))
    return importance

