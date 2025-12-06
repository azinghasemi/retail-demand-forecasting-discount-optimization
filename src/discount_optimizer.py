"""
Discount optimization functions for maximizing revenue.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def calculate_price_elasticity(df, price_col='price', sales_col='sales', discount_col='discount'):
    """
    Calculate price elasticity of demand.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with price and sales data
    price_col : str
        Name of price column
    sales_col : str
        Name of sales column
    discount_col : str
        Name of discount column
        
    Returns:
    --------
    float
        Price elasticity coefficient
    """
    if discount_col in df.columns:
        # Calculate effective price
        df_calc = df.copy()
        df_calc['effective_price'] = df_calc[price_col] * (1 - df_calc[discount_col] / 100)
        
        # Calculate elasticity
        price_changes = df_calc['effective_price'].pct_change()
        sales_changes = df_calc[sales_col].pct_change()
        
        # Remove infinite and NaN values
        valid_mask = (np.isfinite(price_changes)) & (np.isfinite(sales_changes)) & (price_changes != 0)
        elasticity = (sales_changes[valid_mask] / price_changes[valid_mask]).mean()
        
        return elasticity
    else:
        return -1.5  # Default elasticity value


def predict_sales_with_discount(base_sales, discount_pct, elasticity=-1.5):
    """
    Predict sales given a discount percentage.
    
    Parameters:
    -----------
    base_sales : float
        Base sales without discount
    discount_pct : float
        Discount percentage
    elasticity : float
        Price elasticity of demand
        
    Returns:
    --------
    float
        Predicted sales
    """
    # Price change = -discount_pct
    # Sales change = elasticity * price_change
    price_change = -discount_pct / 100
    sales_change = elasticity * price_change
    predicted_sales = base_sales * (1 + sales_change)
    
    return max(0, predicted_sales)  # Ensure non-negative


def calculate_revenue(sales, price, discount_pct):
    """
    Calculate revenue given sales, price, and discount.
    
    Parameters:
    -----------
    sales : float
        Number of units sold
    price : float
        Original price
    discount_pct : float
        Discount percentage
        
    Returns:
    --------
    float
        Total revenue
    """
    effective_price = price * (1 - discount_pct / 100)
    revenue = sales * effective_price
    return revenue


def optimize_discount(base_sales, base_price, elasticity=-1.5, 
                     min_discount=0, max_discount=50, cost_per_unit=None):
    """
    Optimize discount to maximize revenue or profit.
    
    Parameters:
    -----------
    base_sales : float
        Base sales without discount
    base_price : float
        Original price
    elasticity : float
        Price elasticity of demand
    min_discount : float
        Minimum discount percentage
    max_discount : float
        Maximum discount percentage
    cost_per_unit : float
        Cost per unit (if provided, optimizes profit instead of revenue)
        
    Returns:
    --------
    dict
        Dictionary with optimal discount and metrics
    """
    def objective(discount):
        """Objective function to maximize (negative for minimization)."""
        predicted_sales = predict_sales_with_discount(base_sales, discount, elasticity)
        
        if cost_per_unit is not None:
            # Optimize profit
            revenue = calculate_revenue(predicted_sales, base_price, discount)
            cost = predicted_sales * cost_per_unit
            profit = revenue - cost
            return -profit  # Negative because we minimize
        else:
            # Optimize revenue
            revenue = calculate_revenue(predicted_sales, base_price, discount)
            return -revenue  # Negative because we minimize
    
    # Optimize
    result = minimize(
        objective,
        x0=[min_discount + (max_discount - min_discount) / 2],
        bounds=[(min_discount, max_discount)],
        method='L-BFGS-B'
    )
    
    optimal_discount = result.x[0]
    predicted_sales = predict_sales_with_discount(base_sales, optimal_discount, elasticity)
    revenue = calculate_revenue(predicted_sales, base_price, optimal_discount)
    
    if cost_per_unit is not None:
        cost = predicted_sales * cost_per_unit
        profit = revenue - cost
        return {
            'optimal_discount': optimal_discount,
            'predicted_sales': predicted_sales,
            'revenue': revenue,
            'profit': profit,
            'cost': cost
        }
    else:
        return {
            'optimal_discount': optimal_discount,
            'predicted_sales': predicted_sales,
            'revenue': revenue
        }


def create_discount_calendar(df, model, feature_cols, date_col='date',
                            base_price=10.0, elasticity=-1.5, 
                            min_discount=0, max_discount=50):
    """
    Create optimized discount calendar for future dates.
    
    Note: For Season × Product discount calendar, use discount_calendar.py
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with future dates and features
    model : object
        Trained forecasting model
    feature_cols : list
        List of feature column names
    date_col : str
        Name of date column
    base_price : float
        Base price
    elasticity : float
        Price elasticity
    min_discount : float
        Minimum discount
    max_discount : float
        Maximum discount
        
    Returns:
    --------
    pd.DataFrame
        Discount calendar with optimized discounts
    """
    calendar = df.copy()
    
    # Predict base sales
    X = calendar[feature_cols]
    base_sales_pred = model.predict(X)
    calendar['predicted_sales'] = base_sales_pred
    
    # Optimize discount for each row
    optimal_discounts = []
    revenues = []
    
    for idx, row in calendar.iterrows():
        result = optimize_discount(
            base_sales_pred[idx],
            base_price,
            elasticity,
            min_discount,
            max_discount
        )
        optimal_discounts.append(result['optimal_discount'])
        revenues.append(result['revenue'])
    
    calendar['optimal_discount'] = optimal_discounts
    calendar['predicted_revenue'] = revenues
    calendar['predicted_sales_with_discount'] = [
        predict_sales_with_discount(base_sales_pred[i], optimal_discounts[i], elasticity)
        for i in range(len(calendar))
    ]
    
    return calendar

