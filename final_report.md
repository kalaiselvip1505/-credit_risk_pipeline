# Final Report — Credit Risk Model & Interpretability

## 1. Model performance
AUC: 0.5700
F1: 0.0000
Precision: 0.0000
Recall: 0.0000
Confusion matrix:
[[80  0]
 [20  0]]

## 2. SHAP Global Insights
Top features:
- dti
- num_open_accounts
- employment_length
- annual_income
- credit_score

## 3. LIME Local Explanation Summary
All five selected cases were low-risk predictions (0). LIME showed:
- dti, employment_length lowered risk
- num_open_accounts increased risk for some cases

## 4. SHAP vs LIME Comparison
- SHAP gives population-wide trends
- LIME gives per-customer reasoning
- Use SHAP for policy, LIME for individual fairness

## 5. Recommendations
- Fix class imbalance
- Add engineered features
- Re-tune model
