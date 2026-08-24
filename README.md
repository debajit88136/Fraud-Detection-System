# Explainable Real-Time Fraud Detection System

An end-to-end machine learning system that detects fraudulent credit card transactions in real time, explains why each prediction was made using SHAP, and monitors incoming data for statistical drift.

## Why This Project

Most fraud-detection projects stop at "train a model, report accuracy." This one goes further:

1. Explainability - every prediction comes with a SHAP-based breakdown of which features drove the decision.
2. Model comparison - Random Forest and XGBoost were both trained and evaluated; the better model was selected based on F1-score and ROC-AUC.
3. Drift monitoring - a statistical detector flags when incoming transaction patterns diverge from training data.

## Problem: Severe Class Imbalance

The dataset (Kaggle's Credit Card Fraud Detection) contains 284,807 transactions, of which only 0.17% are fraudulent. Accuracy is a meaningless metric here - a model predicting "legitimate" every time would score 99.8% accuracy while catching zero fraud.

This project evaluates using Precision, Recall, F1-score, and ROC-AUC instead, and handles imbalance using class-weighting (Random Forest) and scale_pos_weight (XGBoost).

## Model Comparison

| Model | Precision (Fraud) | Recall (Fraud) | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Random Forest | 0.96 | 0.74 | 0.839 | 0.953 |
| XGBoost (selected) | 0.87 | 0.84 | 0.854 | 0.971 |

XGBoost was selected as the final model. While Random Forest had higher precision, XGBoost caught more actual fraud cases (higher recall) with better overall F1 and ROC-AUC.

## Architecture

Transaction Data -> Data Cleaning & Feature Scaling -> Model Training (RF + XGBoost) -> Comparison -> Best Model Selected -> SHAP Explainer -> FastAPI Backend (/predict, /drift, /logs, /stats) -> Streamlit Dashboard (live feed simulation + manual check)

## How It Works

1. Training (model/train.py): Loads data, scales Amount feature, trains both models, compares them, saves the better one along with a SHAP explainer and reference stats for drift detection.

2. Serving (api/main.py): FastAPI backend loads the model and returns fraud probability, prediction, top 5 SHAP feature contributions, and drift status for each transaction.

3. Drift Detection (api/drift.py): Maintains a rolling window of recent transactions, compares feature means against training-time statistics, flags significant shifts.

4. Dashboard (dashboard/app.py): Streamlit UI with live-feed simulation and manual transaction checking, both with full SHAP explanations.

## Tech Stack

Modeling: scikit-learn, XGBoost, SHAP
Backend: FastAPI, Pydantic, Uvicorn
Frontend: Streamlit
Data: pandas, NumPy
Persistence: joblib

## Project Structure

fraud-detection-project/
- model/train.py - Trains, compares, and saves the model
- api/main.py - FastAPI app
- api/drift.py - Drift detection logic
- api/*.pkl - Saved model artifacts
- dashboard/app.py - Streamlit dashboard

## Running Locally

1. Clone the repository

2. Set up a virtual environment

3. Install dependencies

4. Start the API

5. Start the dashboard (new terminal)

## Evaluation Metrics - Why Not Accuracy

With fraud making up 0.17% of transactions, accuracy is misleading. This project reports Precision, Recall, F1-score, and ROC-AUC instead.

## Possible Extensions

- Replace custom drift detector with a production-grade library (evidently)
- Persist prediction logs to a database instead of in-memory storage
- Add automated retraining trigger when drift is detected
- Add API authentication
- Containerize with Docker and deploy to a cloud service

## Dataset

Credit Card Fraud Detection dataset - anonymized real-world credit card transactions, made available by the Machine Learning Group at ULB on Kaggle.
