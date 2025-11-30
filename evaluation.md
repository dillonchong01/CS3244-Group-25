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

**Key Findings from Random Forest Feature Importance Analysis:**

Based on the Random Forest model's feature importance scores, the following features have the strongest influence on HDB resale flat prices (in descending order of importance):

**Top 5 Most Important Features:**

1. **Floor Area (sqm)** - Highest importance score, strongest predictor of resale price
2. **Remaining Lease Years** - Second most important, showing significant depreciation effect
3. **Storey Range** - Third most important, indicating premium for higher floors
4. **Town/Location** - Strong importance, confirming location-based price variations
5. **Distance to MRT** - Notable importance from engineered features, showing proximity value

The Random Forest model's feature importance analysis reveals that **physical characteristics** (particularly floor area) combined with **temporal factors** (remaining lease) and **location attributes** are the primary drivers of resale flat prices. The ensemble nature of Random Forest allows it to capture complex non-linear interactions between these features effectively.

**Model Evidence:** The feature importance rankings were extracted directly from the trained Random Forest model using the built-in `feature_importances_` attribute, which measures how much each feature contributes to reducing prediction error across all trees in the forest.

### Q2: How does location/flat type/remaining lease years affect the price of the resale flat?

**Analysis based on Lasso Regression Coefficients:**

The Lasso regression model provides interpretable linear coefficients showing the direct impact of each feature on resale prices:

**Location Impact:**
Based on the Lasso coefficient analysis:

- Different towns show significant price premiums or discounts relative to the baseline
- Central and mature estate locations consistently command higher prices
- The magnitude of town coefficients reveals substantial location-based price variations
- Towns with positive coefficients indicate price premiums, while negative coefficients indicate discounts

**Flat Type Impact:**

- Larger flat types (4-room, 5-room, Executive) show positive coefficients
- Each flat type category has a distinct price impact after controlling for floor area
- The coefficients represent the additional value (or discount) associated with each flat type
- Executive flats and larger units command premium prices beyond their floor area

**Remaining Lease Impact:**

- The Remaining Lease Years coefficient shows a positive relationship with price
- Each additional year of remaining lease adds a quantifiable amount to the resale price
- This confirms the depreciation hypothesis: flats with longer remaining leases are valued higher
- The linear coefficient from Lasso provides a clear per-year price impact

**Interpretation of Coefficients:**

- **Positive coefficients**: Features that increase resale price
- **Negative coefficients**: Features that decrease resale price
- **Larger absolute values**: Stronger impact on price
- **Zero coefficients**: Features removed by Lasso (less important for prediction)

The Lasso model selected only the most impactful features, automatically performing feature selection through L1 regularization. This provides a clearer picture of which specific locations, flat types, and lease-related factors have the strongest direct effects on pricing.

### Q3: What is the expected price of the resale flat in x years?

**Analysis Methodogy**
To predict future HDB prices, we first trained two models (linear regression and random forest) on past data (2017-early 2024) and predicted on 'future' data (late 2024-2025). 

The two models were evaluated, the performance metrics are as follows:
- Linear Regression:
  - MAE: 60805.22
  - RMSE: 80252.47
  - R2: 0.747

- Random Forest:
  - MAE: 53219.98
  - RMSE: 68633.35
  - R2: 0.814

Random Forest performs better at predicting future prices. We proceeded with illustratng the prediction of future prices (2025-2045) using the Random Forest model. Visualisations was done with flats that start off with a 99 year lease in 2025, to simulate the resale prices of newly-bought flats in the next 20 years. 

**Configuration Design**
Representative flat configurations were created across multiple dimensions:
1. Flat Type. Floor area was used to represent each flat type: 1 Room (30 sqm) to 3Gen/Executive flats(160 sqm)
2. Storey level. Low (5th floor), Medium (15th floor) and High (30th floor)
3. Remaining lease. Short (50 years), Medium (70 years) and Long (99 years)

This resulted in a total of 54 combinations of flats

**Temporal Feature Adjustment**
'Year' and 'Remaining_Lease' were adjusted accordingly for each year ahead (0-20 years)

**Key Assumptions**
1. Flats have been categorised by storey and floor area (1, 2, 3, 4 and 5 Room flats)
2. Price depreciation follows the patterns observed in historical data, which has been used to train the RF model
3. Market trends continue
4. Example flats are generated with mean distances to MRT and malls from the data set, and are in non-mature estates, within 1km of a primary school

**Limitations**
- Model reliability decreases with increasing years into the future, wherer extrapolation from training increases

**Findings**
1. Floor area emerges as the dominant factor in both current valuation and future price trajectories, whereas storey levels have a modest impact on prices. 
- **At Year = 0 (2025)**, 
| Flat Type | Mean Price |
|-----------|------------|
| 1-Room | $273,063 |
| 2-Room | $337,245 |
| 3-Room | $495,800 |
| 4-Room | $528,965 |
| 5-Room | $655,943 |

- Clear price heirarchy based on flat size. 5 room flats commanding highest prices (>$650k) and 1-room flats commanding the lowest prices (<$275k).
- Price increases are non-linear with size. The jump from 4 room to 5 room ($127k) is much greater than the price jump from 1 room to 2 room flats ($64k)

| Storey Level | Mean Price | 
|--------------|------------|
| Low (5th floor) | $502,703 |
| Mid (15th floor) | $531,957 |
| High (30th floor) | $533,454 |

- Prices deviate about $30k between low and medium storeys, and the small difference between medium and high storeys suggests diminishing returns for very high floors. 

| Remaining Lease | Mean Price |
|-----------------|------------|
| Short (50 years) | $468,502 |
| Medium (70 years) | $509,709 |
| Long (99 years) | $589,903 |

- There is a strong positive correlation between remaining lease and price. Long lease commands on average $121k more than short lease, suggesting that lease periods greatly affects market value.

**Key Insight**: 
- While higher storeys are more valuable, the effect is much smaller than floor area. Hence for investment purposes, flat size matters more than storey level

Insignts from visualizaition:
- Flat prices are generally stagnant from 0-5 years ahead, likely due to the 5 year minimum occupation period (MOP) of flats. 
- All flat types experience a sharp drop around the year 12, which may be a reflection of lease decay acceleration.
- Price stability differs:
  - 5 room flats remain the most resilient after the major dip whereas 4 and 3 room flats continue to decline

**Combined Effects: Optimal Configurations**
Based on flat size and storey height, the model predicts that larger flats on higher floors provide the strongest 10-year performance.  
Top performing combinations are:
- 5 room, High Storey
  - Current ~$810k to **~$770k** in 10 years

- 4 room, High/Mid Storey
  - Current ~$650k to **~$630k** in 10 years

These flats perfom well as the medium/larger sizes are more appealing, with a larger resale market for such flats. The storey premium of higher storey flats is also able to offset lease decay


### Q4: What type of resale unit and duration allows the owner to obtain the largest profit upon selling?

**Analysis Methodology:**

To answer this question, we used the XGBoost model (our best-performing model) to simulate future resale prices and calculate potential profits for different unit types and holding durations. Due to data limitations on original purchase prices, we assumed owners purchased their units as resale flats and calculated profit as: **Predicted Resale Value - Initial Purchased Resale Value**.

**Key Assumptions:**

1. Minimum resaleable lease set at 20 years (below which resale becomes impractical)
2. Holding durations analyzed: 1-15 years
3. Depreciation effects already captured in the XGBoost model training data (2017-2024)
4. Unit types categorized by: Estate maturity, Storey level, and Floor area

**Findings:**

**Optimal Holding Duration:**

The analysis reveals that **8 years** is the optimal holding period for maximizing profit. After this point, price appreciation plateaus, suggesting diminishing returns from longer holding periods.

**Estate Maturity Impact:**

- **Non-mature estates** consistently yield higher profits compared to mature estates across all holding durations
- This suggests stronger appreciation potential in developing areas with improving infrastructure

**Storey Level Impact:**

The middle-range storey levels perform best:

1. **11-15 storeys** and **16-20 storeys**: Highest profit margins
2. **Lower levels (1-3, 4-6, 7-10)**: Moderate profits
3. **High levels (21-30, 30+)**: Lowest profit margins

This indicates strong market preference for mid-level units that balance accessibility with views.

**Floor Area Impact:**

Analyzing profit percentage by floor area reveals:

- **80-120 sqm units**: Highest profit percentages (~34-35% after 10 years)
  - Optimal size range for families
  - Strong resale demand
  - Best balance of affordability and spaciousness
- **<60 sqm units**: Lower profit percentages (~29% max)
  - Lower base values limit absolute appreciation
  - Smaller market segment
- **>160 sqm units**: Lowest returns (~19%)
  - Luxury/niche segment with limited buyer pool
  - High absolute prices but lower percentage returns

**Optimal Investment Profile:**

To maximize profit upon resale, homeowners should target:

- **Holding Duration:** 8 years
- **Estate Type:** Non-mature estates
- **Floor Area:** 80-120 sqm
- **Storey Level:** 11-15 storeys

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
