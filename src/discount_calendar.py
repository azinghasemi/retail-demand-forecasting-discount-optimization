"""
Discount calendar creation based on Season × Product combinations.
This matches the methodology described in the report.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def create_season_product_discount_calendar(df, model, feature_cols, 
                                           product_col='Product ID',
                                           season_col='Season',
                                           discount_col='Discount',
                                           price_col='Price',
                                           demand_col='Demand'):
    """
    Create discount calendar for Season × Product combinations.
    
    This function implements the methodology from the report:
    1. For each Product × Season combination
    2. Test multiple discount scenarios (0%, 5%, 10%, ..., 25%)
    3. Predict demand for each scenario
    4. Calculate revenue = pred_demand * price * (1 - disc / 100)
    5. Select discount that maximizes revenue
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with grocery sales data
    model : object
        Trained XGBoost model for demand prediction
    feature_cols : list
        List of feature column names used by the model
    product_col : str
        Name of product ID column
    season_col : str
        Name of season column
    discount_col : str
        Name of discount column
    price_col : str
        Name of price column
    demand_col : str
        Name of demand column
        
    Returns:
    --------
    pd.DataFrame
        Discount calendar with optimal discounts for each Product × Season
    """
    # Get unique products and seasons
    products = df[product_col].unique()
    
    # Map months to seasons if season column doesn't exist
    if season_col not in df.columns and 'Month' in df.columns:
        season_map = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        }
        df['Season'] = df['Month'].map(season_map)
        season_col = 'Season'
    
    seasons = df[season_col].unique()
    
    # Get discount levels from data (0%, 5%, 10%, ..., 25%)
    if discount_col in df.columns:
        discount_levels = sorted(df[discount_col].unique())
    else:
        discount_levels = [0, 5, 10, 15, 20, 25]
    
    records = []
    
    # Iterate through each Product × Season combination
    for pid in products:
        for season in seasons:
            # Get subset for this product and season
            subset = df[(df[product_col] == pid) & (df[season_col] == season)]
            
            if subset.empty:
                continue
            
            # Get median values for base features
            base = subset.median(numeric_only=True).to_dict()
            price = float(subset[price_col].median())
            
            best_revenue = -np.inf
            best_discount = 0
            best_predicted_demand = 0
            
            # Test each discount scenario
            for disc in discount_levels:
                # Initialize feature dictionary
                feat = {col: 0 for col in feature_cols}
                
                # Fill base numeric features
                for k, v in base.items():
                    if k in feat:
                        feat[k] = v
                
                # Override discount
                feat[discount_col] = disc
                
                # Set one-hot encoded features for product and season
                prod_col_name = f"{product_col}_{pid}"
                if prod_col_name in feat:
                    feat[prod_col_name] = 1
                
                season_col_name = f"{season_col}_{season}"
                if season_col_name in feat:
                    feat[season_col_name] = 1
                
                # Create feature vector
                X_scenario = pd.DataFrame([feat])
                
                # Ensure all feature columns are present
                missing_cols = set(feature_cols) - set(X_scenario.columns)
                for col in missing_cols:
                    X_scenario[col] = 0
                
                X_scenario = X_scenario[feature_cols]
                
                # Predict demand
                pred_demand = model.predict(X_scenario)[0]
                
                # Calculate revenue: revenue = pred_demand * price * (1 - disc / 100)
                revenue = pred_demand * price * (1 - disc / 100.0)
                
                # Track best scenario
                if revenue > best_revenue:
                    best_revenue = revenue
                    best_discount = disc
                    best_predicted_demand = pred_demand
            
            # Save best scenario for this Product × Season
            records.append({
                product_col: pid,
                season_col: season,
                'Optimal_Discount': best_discount,
                'Predicted_Demand': best_predicted_demand,
                'Price': price,
                'Estimated_Revenue': best_revenue
            })
    
    # Create calendar dataframe
    calendar = pd.DataFrame(records)
    
    return calendar


def create_discount_calendar_heatmap(calendar, product_col='Product ID', 
                                     season_col='Season', 
                                     discount_col='Optimal_Discount',
                                     save_path=None):
    """
    Create a heatmap visualization of the discount calendar.
    
    Parameters:
    -----------
    calendar : pd.DataFrame
        Discount calendar dataframe
    product_col : str
        Name of product column
    season_col : str
        Name of season column
    discount_col : str
        Name of discount column
    save_path : str
        Path to save the heatmap image
        
    Returns:
    --------
    matplotlib figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create pivot table for heatmap
    pivot = calendar.pivot_table(
        values=discount_col,
        index=product_col,
        columns=season_col,
        aggfunc='mean'
    )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot) * 0.5)))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', 
                cbar_kws={'label': 'Discount (%)'}, ax=ax)
    ax.set_title('Product × Seasonal Discount Calendar', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Product ID', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    
    # Display the figure
    plt.show()
    return fig

