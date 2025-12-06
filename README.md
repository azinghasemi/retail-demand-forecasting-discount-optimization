# Retail Demand Forecasting & Discount Optimization

An end-to-end machine learning project using historical retail data (2022–2024)

**Author:** Azin Ghasemi

**Programme:** Fundamentals of Data Analytics (MSc)

**Year:** 2025

## Project Overview

This project builds a demand forecasting system for grocery products across five retail stores using sales data from 2022–2024.

The goal is to:

- Forecast daily demand
- Understand seasonal patterns
- Build an optimal discount calendar to increase revenue and reduce food waste
- Provide actionable insights for retail store decision-making

The final solution includes:

- Exploratory Data Analysis
- Feature engineering (including a custom "Interpolated Order" variable)
- Multiple ML models: Linear Regression, Random Forest, XGBoost
- Hyperparameter tuning
- Scenario-based discount optimization
- A final Product × Season discount calendar

Full academic report is available in `/reports/FDA_AZIN_GHASEMI_SE.pdf`.

## Repository Structure

```
.
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_xgboost_tuning.ipynb
│   ├── 06_discount_calendar.ipynb
│   └── 07_business_insights.ipynb
│
├── src/
│   ├── data_prep.py
│   ├── feature_engineering.py
│   ├── eda_plots.py
│   ├── model_linear_regression.py
│   ├── model_random_forest.py
│   ├── model_xgboost.py
│   ├── evaluation.py
│   └── discount_optimizer.py
│
├── reports/
│   ├── FDA_AZIN_GHASEMI_SE.pdf
│   └── figures/
│
├── requirements.txt
├── README.md
└── LICENSE
```

## Dataset Description

The dataset contains daily sales information for 5 stores and 5 product categories, with this project focusing on Groceries.

According to the variable table (page 13 of the report), the dataset includes:

- Date
- Store & Product identifiers
- Unit Sold
- Demand
- Inventory Level
- Price
- Competitor Price
- Promotion / Discount flags
- Epidemic flag (COVID impact)

The grocery category contains 30,400 rows and 16 features.

## Exploratory Data Analysis

The EDA phase includes:

**Histograms of all numeric variables (p.15)**
- Used to detect skewness and decide on normalization.

**Correlation Heatmap (p.20)**
- Shows strongest correlations with Demand, helping with feature selection.

**Seasonal & Monthly Patterns (p.21–24)**
- Heatmaps revealed:
  - Strong seasonality in some products
  - Large demand shifts during epidemic months
  - Monthly trends differ between 2022 and 2023
  - Epidemic effect is a critical forecasting feature

**Epidemic Impact (p.23–24)**
- A major cause of forecast error and deviation in seasonal patterns.

## Feature Engineering

**Time-based features (p.25)**
- Created using Pandas:
  - Year
  - Month
  - Week
  - Day
  - Day of week
  - Is weekend
  - Is month start / end
  - Quarter

**Interpolated Order (custom feature) (p.17–18)**
- Because "Units Ordered" had many zeros, the following was created:
  - `Interpolated_Order = Last_Order_Amount / Days_Since_Last_Order`
- This helped reduce sparsity and improved model performance.

**Holiday Encoding**
- Using the US Federal Holiday calendar.

## Models Implemented

Three models were trained and compared, each using various feature sets and split strategies:

### 1. Linear Regression
- Best R²: 74% using time-based split and all features. (p.35)

### 2. Random Forest
- Best R²: 83% using 5-fold CV and full feature set. (p.35–36)

### 3. XGBoost (Best Model)
- Best R²: 90%
- Using:
  - Feature Set 2
  - Time split
  - Hyperparameter configuration xgb_3 (p.32, 36)
- XGBoost's robustness to non-linear patterns and mixed feature types made it the top performer.

## Error Analysis

Monthly error heatmap (p.37) showed:

- Highest errors occur during epidemic periods
- Suggesting epidemic flags are essential features
- Models must account for extreme irregularity

## Discount Optimization Engine

The discount engine performs:

1. Generate discount scenarios (0%, 5%, 10%, … 25%)
2. Predict demand for each discount
3. Calculate revenue using:
   - `revenue = pred_demand * price * (1 - discount / 100)`
4. Select the discount that produces maximum revenue
5. Build a Season × Product discount calendar

Final calendar visual is shown on page 40 of the report.

## Final Output: Discount Calendar

The result is a heatmap showing the recommended discount for:

- 19 grocery products
- Across Spring, Summer, Fall, Winter

Example interpretation:

- Some products need 10–25% discount in low-demand seasons
- Others should avoid discounting during peak periods
- Epidemic seasons change optimal discount levels

The CSV and heatmap image are included in the repo.

## Key Business Insights

From the discussion section (p.41):

- Demand forecasting helps reduce food waste
- Seasonal discounting stabilizes inventory levels
- Low-demand products should be bundled with high-demand ones
- Forecasting improves reorder point calculation
- Epidemic conditions materially affect demand and pricing strategy

## Limitations & Future Work

(p.41–42 of the report)

- Only two years of data → limited seasonality learning
- Epidemic periods distort model stability

Future work:

- Train models separately for epidemic vs non-epidemic periods
- Improve price-based revenue models
- Extend dataset for stronger forecasting patterns

## How to Run the Project

```bash
git clone https://github.com/<your-username>/retail-demand-forecasting-discount-optimization
cd retail-demand-forecasting-discount-optimization
pip install -r requirements.txt
jupyter notebook
```

Open the notebooks in order (01 → 07).

## Skills Demonstrated

- Time-series forecasting
- Feature engineering
- Advanced ML modeling (Random Forest, XGBoost)
- Hyperparameter tuning
- Error analysis
- Business analytics & retail strategy
- Scenario simulation & optimization
- Data visualization
- Report writing & interpretation

## License

MIT License
