# -credit_risk_pipeline
Credit Risk Model with SHAP, LIME, and XGBoost - Explainable AI Project
# Credit Risk Prediction — SHAP & LIME Interpretability

## Project overview
This repository contains a credit risk prediction pipeline (XGBoost) together with global (SHAP) and local (LIME) interpretability outputs. The goal is to predict loan default and provide both population-level feature importance and case-level explanations for borderline decisions.

## Contents (important files)
- `credit_risk_full_code.py` — main runnable pipeline (training, SHAP, LIME).
- `model_metrics.txt` — test metrics summary.
- `final_report.md` — full interpretability report (global vs local explanations & policy implications).
- `interpretation_report.txt` — detailed text report generated during run. (local path: `/mnt/data/interpretation_report.txt`)
- `lime_local_explanations.csv` — local LIME explanations CSV. (local path: `/mnt/data/lime_local_explanations.csv`)
- `final_model.pkl` — saved XGBoost model. (local path: `/mnt/data/final_model.pkl`)
- `shap_sample.npz` — saved SHAP values sample. (local path: `/mnt/data/shap_sample.npz`)
- `images/` — SHAP and LIME PNGs (if present).

> **Note:** Files referenced with `/mnt/data/...` are the artifacts created in your Colab environment — download them from Colab and upload to this repo (see Upload instructions below).

## Quick stats (from latest run)
See `model_metrics.txt` for full numbers. Short summary:
- AUC: 0.5700
- F1: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- Confusion matrix: `[[80  0] [20  0]]`

## How to reproduce (Colab / local)
1. Place your dataset CSV in the repo (or in Colab workspace). Update `DATA_FILE` path in `credit_risk_full_code.py` if needed.
2. Install dependencies:
```bash
pip install -r requirements.txt
# or individually:
pip install pandas numpy scikit-learn xgboost shap lime matplotlib joblib
