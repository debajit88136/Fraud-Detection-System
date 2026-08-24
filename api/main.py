from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List
import joblib
import numpy as np
import pandas as pd

from drift import DriftDetector

app = FastAPI(title="Fraud Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load saved model artifacts
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
explainer = joblib.load("shap_explainer.pkl")
reference_stats = joblib.load("reference_stats.pkl")
feature_columns = joblib.load("feature_columns.pkl")

drift_detector = DriftDetector(reference_stats, feature_columns)

prediction_log: List[dict] = []


class Transaction(BaseModel):
    features: Dict[str, float] = Field(
        ..., description="Dictionary of feature_name: value, matching training columns"
    )


@app.get("/")
def root():
    return {"status": "Fraud Detection API is running", "version": "1.0"}


@app.post("/predict")
def predict(transaction: Transaction):
    try:
        row = {col: transaction.features.get(col, 0.0) for col in feature_columns}
        df_row = pd.DataFrame([row])[feature_columns]

        df_row["Amount"] = scaler.transform(df_row[["Amount"]])

        proba = model.predict_proba(df_row)[0, 1]
        prediction = int(proba >= 0.5)

        shap_values = explainer.shap_values(df_row)
        fraud_shap = shap_values[0][:, 1] if hasattr(shap_values[0], 'shape') and len(shap_values[0].shape) > 1 else shap_values[1][0]
        contributions = dict(zip(feature_columns, [round(float(v), 4) for v in fraud_shap]))
        top_reasons = dict(
            sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        )

        drift_detector.add_transaction(row)
        drift_status = drift_detector.check_drift()

        result = {
            "fraud_probability": round(float(proba), 4),
            "prediction": "FRAUD" if prediction == 1 else "LEGITIMATE",
            "top_contributing_features": top_reasons,
            "drift_status": drift_status,
        }

        prediction_log.append(result)
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/logs")
def get_logs(limit: int = 50):
    return prediction_log[-limit:]


@app.get("/drift")
def get_drift_status():
    return drift_detector.check_drift()


@app.get("/stats")
def get_stats():
    if not prediction_log:
        return {"total_predictions": 0, "fraud_rate": 0}
    total = len(prediction_log)
    fraud_count = sum(1 for p in prediction_log if p["prediction"] == "FRAUD")
    return {
        "total_predictions": total,
        "fraud_count": fraud_count,
        "fraud_rate": round(fraud_count / total, 4),
    }
