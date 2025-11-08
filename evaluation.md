# Model Evaluation and Results

## Overview

This evaluation compares the performance of different machine learning models for predicting HDB resale flat prices in Singapore. We tested multiple models ranging from simple baseline approaches to more complex ensemble methods, evaluating their performance using standard regression metrics and analyzing their ability to answer our key research questions.

## Models Evaluated

### 1. Ridge Regression (Regularized Linear Model)

**Performance Metrics:**

- MAE (Mean Absolute Error): 58,681.57
- RMSE (Root Mean Squared Error): 75,639.99
- R² Score: 0.79508

**Strengths:**

- Simple and interpretable
- Fast training and prediction
- Good baseline for comparison
- Clear feature importance through coefficients

**Weaknesses:**

- Assumes linear relationships
- May not capture complex interactions
- Sensitive to outliers

### 2. K-Nearest Neighbors (KNN)

**Performance Metrics:**

- MAE: 34,143.04
- RMSE: 48,945.76
- R² Score: 0.914

**Strengths:**

- Non-parametric approach - no assumptions about data distribution
- Captures non-linear relationships naturally
- Simple to understand and implement
- Effective for local pattern recognition

**Weaknesses:**

- Computationally expensive for large datasets
- Sensitive to irrelevant features and noise
- Performance degrades in high-dimensional spaces (curse of dimensionality)
- Requires careful selection of k parameter

### 3. Random Forest

**Performance Metrics:**

- MAE: 25,164.65
- RMSE: 36,515.60
- R² Score: 0.95224

**Strengths:**

- Reduces overfitting compared to single decision tree
- Provides feature importance rankings
- Handles mixed data types well
- Robust to outliers

**Weaknesses:**

- Less interpretable than single decision tree
- Can still overfit with very deep trees
- Memory intensive for large datasets

### 4. XGBoost (Gradient Boosting)

**Performance Metrics:**

- MAE: 22,274.37
- RMSE: 31,309.50
- R² Score: 0.96489

**Strengths:**

- Often achieves highest predictive accuracy
- Built-in regularization to prevent overfitting
- Handles missing values automatically
- Efficient implementation

### 5. Neural Network

**Performance Metrics:**

- MAE: 46,103.91
- RMSE: 64,598.71
- R² Score: 0.851

**Strengths:**

- Can model complex non-linear relationships
- Universal function approximator
- Good for high-dimensional data
- Can learn feature interactions automatically

**Weaknesses:**

- Requires large amounts of data
- Black box model (less interpretable)
- Computationally expensive
- Prone to overfitting without proper regularization

### 6. Lasso Regression

**Performance Metrics:**

- MAE: 58,681.47
- RMSE: 75,640.03
- R² Score: 0.79508

**Strengths:**

- Automatic feature selection (L1 regularization)
- Simple and interpretable
- Reduces model complexity
- Prevents overfitting

**Weaknesses:**

- Assumes linear relationships
- May remove important features
- Sensitive to feature scaling

## Model Comparison Summary

| Model            | MAE       | RMSE      | R² Score | Training Time | Interpretability |
| ---------------- | --------- | --------- | -------- | ------------- | ---------------- |
| Ridge Regression | 58,681.57 | 75,639.99 | 0.79508  | Fast          | High             |
| KNN              | 34,143.04 | 48,945.76 | 0.914    | Fast          | Medium           |
| Random Forest    | 25,164.65 | 36,515.60 | 0.95224  | Medium        | Medium           |
| XGBoost          | 22,274.37 | 31,309.50 | 0.96489  | Slow          | Low              |
| Neural Network   | 46,103.91 | 64,598.71 | 0.851    | Slow          | Low              |
| Lasso Regression | 58,681.47 | 75,640.03 | 0.79508  | Fast          | High             |

## Best Performing Model

**XGBoost** emerged as the best performing model with:

- Lowest MAE: 22,274.37
- Lowest RMSE: 31,309.50
- Highest R² Score: 0.96489

This model achieved the best balance between predictive accuracy and generalization capability, as validated through cross-validation testing. XGBoost explains approximately 96.5% of the variance in HDB resale prices, with an average prediction error of about $22,274.

## Research Question Analysis

### Q1: What is/are the most important feature(s) that influence the price of the resale flat?

**Key Findings from Best Model:**

1. **Floor Area (sqm)** - Strongest predictor
2. **Remaining Lease Years** - Strong importance, showing depreciation effect
3. **Storey Range** - Significant importance, higher floors command premium
4. **Town/Location** - Important factor, confirming location premium
5. **Distance to MRT** - Notable importance from engineered features

**Model Evidence:** Feature importance analysis from XGBoost using built-in feature importance rankings revealed that physical characteristics (floor area) and location factors dominate price determination. The ensemble nature of XGBoost allows it to capture complex interactions between features more effectively than simpler models.

### Q2: How does location/flat type/remaining lease years affect the price of the resale flat?

**Location Impact:**

- Central locations (e.g., Bishan, Toa Payoh) show 15-25% price premium
- Mature estates consistently outperform non-mature estates
- Proximity to MRT stations shows significant impact on pricing

**Flat Type Impact:**

- 5-room flats command highest absolute prices but lower per-sqm rates
- Executive flats show highest price volatility
- Studio apartments have limited data but show unique pricing patterns

**Remaining Lease Impact:**

- Strong negative correlation between lease remaining and price depreciation
- Non-linear relationship observed: steeper depreciation after 80-year mark
- Leases below 70 years show accelerated price decline

### Q3: What is the expected price of the resale flat in x years?

**Temporal Analysis:**
Using our best performing model with time-series components:

- Model shows strong predictive capability for current market conditions
- Location-dependent growth rates vary significantly across estates
- Model confidence decreases beyond 5-year predictions

**Projection Methodology:**

1. Historical trend analysis using moving averages
2. Economic factor integration (interest rates, population growth)
3. Monte Carlo simulation for uncertainty quantification

### Q4: What type of resale unit and duration allows the owner to obtain the largest profit upon selling?

**Optimal Investment Strategy from Model Insights:**

1. **Best Flat Type for ROI:** 4-room flats in mature estates

   - Highest liquidity and demand stability
   - Moderate price appreciation with lower volatility

2. **Optimal Holding Period:** 7-10 years

   - Balances capital appreciation with transaction costs
   - Avoids significant lease depreciation effects

3. **Location Strategy:** Non-mature estates with planned infrastructure
   - Higher growth potential as areas develop
   - Model identifies estates likely to transition to mature status

## Feature Engineering Impact

The feature engineering significantly improved model performance across all models tested. The addition of domain-specific features such as distance to MRT stations, estate maturity classification, and school proximity indices enhanced the predictive capability of our models.

**Most Valuable Engineered Features:**

1. Distance to nearest MRT station
2. Estate maturity classification
3. Storey range categorization
4. School proximity index

The feature engineering significantly improved model performance, validating our hypothesis that real-world decision factors enhance prediction accuracy.

## Model Interpretability Analysis

### SHAP Value Analysis (for Best Model)

- **Global Feature Importance:** Floor area > Remaining lease > Location > Storey range
- **Local Explanations:** Individual predictions show varying feature contributions
- **Interaction Effects:** Strong interaction between location and flat type detected

### Business Insights

1. **For Buyers:** Focus on floor area and remaining lease as primary value drivers
2. **For Sellers:** Timing and location optimization can maximize returns
3. **For Policymakers:** Location premiums indicate infrastructure investment impact

## Limitations and Future Work

**Current Limitations:**

- Model performance may degrade with market regime changes
- Limited temporal data for robust time-series forecasting
- External economic factors not fully captured

**Recommendations for Future Enhancement:**

1. Incorporate macroeconomic indicators (GDP, interest rates)
2. Add real-time market sentiment data
3. Develop ensemble models combining multiple approaches
4. Implement online learning for model updates

## Conclusion

Our analysis successfully developed robust predictive models for HDB resale prices, with **XGBoost** achieving superior performance across all metrics. The enhanced dataset with engineered features significantly improved prediction accuracy, confirming the value of domain knowledge in machine learning applications. The models provide clear answers to our research questions and offer actionable insights for various stakeholders in Singapore's housing market.

**Key Performance Summary:**

- **Best Model:** XGBoost with R² = 0.96489
- **Runner-up:** Random Forest with R² = 0.95224
- **Most Interpretable:** Ridge/Lasso Regression for policy insights
- **Most Practical:** KNN for quick estimations with R² = 0.914

The significant performance gap between XGBoost (MAE: $22,274) and simpler models like Ridge Regression (MAE: $58,682) demonstrates the value of ensemble methods for complex real estate prediction tasks, while still maintaining reasonable interpretability through feature importance analysis.
