"""
Random Forest model for demand forecasting.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def train_random_forest(X, y, test_size=0.2, random_state=42, 
                       n_estimators=100, max_depth=10, min_samples_split=5):
    """
    Train Random Forest model.
    
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
    n_estimators : int
        Number of trees
    max_depth : int
        Maximum depth of trees
    min_samples_split : int
        Minimum samples to split
        
    Returns:
    --------
    tuple
        (model, X_train, X_test, y_train, y_test)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print("Random Forest model trained successfully")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Number of trees: {n_estimators}, Max depth: {max_depth}")
    
    return model, X_train, X_test, y_train, y_test


def predict_random_forest(model, X):
    """
    Make predictions using Random Forest model.
    
    Parameters:
    -----------
    model : RandomForestRegressor
        Trained model
    X : pd.DataFrame or np.array
        Feature matrix
        
    Returns:
    --------
    np.array
        Predictions
    """
    predictions = model.predict(X)
    return predictions


def get_feature_importance(model, feature_names):
    """
    Get feature importance from Random Forest.
    
    Parameters:
    -----------
    model : RandomForestRegressor
        Trained model
    feature_names : list
        List of feature names
        
    Returns:
    --------
    dict
        Dictionary with feature names and importance scores
    """
    importance = dict(zip(feature_names, model.feature_importances_))
    # Sort by importance
    importance = dict(sorted(importance.items(), 
                            key=lambda x: x[1], reverse=True))
    return importance

