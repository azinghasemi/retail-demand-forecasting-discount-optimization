"""
XGBoost model for demand forecasting.
"""

import pandas as pd
import numpy as np
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def train_xgboost(X, y, test_size=0.2, random_state=42,
                 n_estimators=100, max_depth=6, learning_rate=0.1):
    """
    Train XGBoost model.
    
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
        Number of boosting rounds
    max_depth : int
        Maximum depth of trees
    learning_rate : float
        Learning rate
        
    Returns:
    --------
    tuple
        (model, X_train, X_test, y_train, y_test)
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    
    # Train model
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print("XGBoost model trained successfully")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Boosting rounds: {n_estimators}, Max depth: {max_depth}, Learning rate: {learning_rate}")
    
    return model, X_train, X_test, y_train, y_test


def predict_xgboost(model, X):
    """
    Make predictions using XGBoost model.
    
    Parameters:
    -----------
    model : XGBRegressor
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
    Get feature importance from XGBoost.
    
    Parameters:
    -----------
    model : XGBRegressor
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


def tune_xgboost_hyperparameters(X, y, param_grid=None, cv=5, random_state=42):
    """
    Tune XGBoost hyperparameters using grid search.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.array
        Feature matrix
    y : pd.Series or np.array
        Target variable
    param_grid : dict
        Parameter grid for grid search
    cv : int
        Number of cross-validation folds
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    XGBRegressor
        Best model
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
    
    from sklearn.model_selection import GridSearchCV
    
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    
    model = xgb.XGBRegressor(random_state=random_state, n_jobs=-1)
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {-grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

