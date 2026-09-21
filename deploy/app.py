import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import time
from collections import deque

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# ---- Load all artifacts once ----
@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    explainer = joblib.load("shap_explainer.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    reference_stats = joblib.load("reference_stats.pkl")
    return model, scaler, explainer, feature_columns, reference_stats

model, scaler, explainer, feature_columns, reference_stats = load_artifacts()

# ---- Drift Detector class ----
class DriftDetector:
    def __init__(self, reference_stats, feature_columns, window_size=200, threshold=3.0):
        self.ref_mean = reference_stats['mean']
        self.ref_std = reference_stats['std']
        self.feature_columns = feature_columns
        self.window = deque(maxlen=window_size)
        self.threshold = threshold

    def add_transaction(self, feature_dict):
        self.window.append(feature_dict)

    def check_drift(self):
        if len(self.window) < 30:
            return {"drift_detected": False, "reason": "not enough data yet", "flagged_features": {}}
        flagged = {}
        for col in self.feature_columns:
            if col not in self.ref_mean:
                continue
            live_values = [t[col] for t in self.window if col in t]
            if not live_values:
                continue
            live_mean = np.mean(live_values)
            ref_mean = self.ref_mean[col]
            ref_std = self.ref_std.get(col, 1e-6) or 1e-6
            z_shift = abs(live_mean - ref_mean) / ref_std
            if z_shift > self.threshold:
                flagged[col] = round(float(z_shift), 3)
        return {
            "drift_detected": len(flagged) > 0,
            "flagged_features": flagged,
            "window_size": len(self.window)
        }

if "drift_detector" not in st.session_state:
    st.session_state.drift_detector = DriftDetector(reference_stats, feature_columns)

if "prediction_log" not in st.session_state:
    st.session_state.prediction_log = []

# ---- Prediction function (replaces the API call) ----
def predict_transaction(row_dict):
    row = {col: row_dict.get(col, 0.0) for col in feature_columns}
    df_row = pd.DataFrame([row])[feature_columns]
    df_row["Amount"] = scaler.transform(df_row[["Amount"]])

    proba = model.predict_proba(df_row)[0, 1]
    prediction = int(proba >= 0.5)

    shap_values = explainer.shap_values(df_row)
    if isinstance(shap_values, list):
        fraud_shap = shap_values[1][0]
    else:
        fraud_shap = shap_values[0]
        if len(fraud_shap.shape) > 1:
            fraud_shap = fraud_shap[:, 1]

    contributions = dict(zip(feature_columns, [round(float(v), 4) for v in fraud_shap]))
    top_reasons = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5])

    st.session_state.drift_detector.add_transaction(row)
    drift_status = st.session_state.drift_detector.check_drift()

    result = {
        "fraud_probability": round(float(proba), 4),
        "prediction": "FRAUD" if prediction == 1 else "LEGITIMATE",
        "top_contributing_features": top_reasons,
        "drift_status": drift_status,
    }
    st.session_state.prediction_log.append(result)
    return result

# ---- UI ----
st.title("🔍 Explainable Real-Time Fraud Detection System")
st.caption("Random Forest / XGBoost model + SHAP explainability + drift monitoring")

tab1, tab2 = st.tabs(["Live Feed Simulation", "Manual Transaction Check"])

with tab1:
    st.subheader("Simulated Real-Time Transaction Feed")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        n_transactions = st.slider("Number of transactions to stream", 5, 100, 20)
        speed = st.slider("Delay between transactions (sec)", 0.0, 2.0, 0.3)
        start_button = st.button("Start Simulation")

    feed_placeholder = st.empty()
    chart_placeholder = st.empty()
    drift_placeholder = st.empty()

    if start_button:
        try:
            sample_df = pd.read_csv("sample_transactions.csv").drop(columns=["Class"], errors="ignore")
        except FileNotFoundError:
            st.error("sample_transactions.csv not found.")
            sample_df = pd.DataFrame()

        fraud_flags = []
        for i in range(min(n_transactions, len(sample_df))):
            row = sample_df.iloc[i].to_dict()
            result = predict_transaction(row)
            fraud_flags.append(1 if result["prediction"] == "FRAUD" else 0)

            with feed_placeholder.container():
                st.write(f"**Transaction #{i+1}**")
                if result["prediction"] == "FRAUD":
                    st.error(f"🚨 FLAGGED AS FRAUD — probability: {result['fraud_probability']:.2%}")
                    st.write("Top contributing features:", result["top_contributing_features"])
                else:
                    st.success(f"✅ Legitimate — fraud probability: {result['fraud_probability']:.2%}")

                drift = result.get("drift_status", {})
                if drift.get("drift_detected"):
                    drift_placeholder.warning(f"⚠️ Data drift detected: {drift.get('flagged_features')}")
                else:
                    drift_placeholder.info("No significant drift detected.")

            running_fraud_rate = pd.Series(fraud_flags).expanding().mean()
            chart_placeholder.line_chart(running_fraud_rate, height=200)
            time.sleep(speed)

        st.success(f"Simulation complete — processed {len(fraud_flags)} transactions.")

with tab2:
    st.subheader("Check a Single Transaction Manually")
    uploaded_file = st.file_uploader("Upload a single-transaction CSV", type=["csv"])

    if uploaded_file is not None:
        row_df = pd.read_csv(uploaded_file)
        row_df = row_df.drop(columns=["Class"], errors="ignore")
        st.write("Preview:", row_df.head(1))

        if st.button("Predict this transaction"):
            row_dict = row_df.iloc[0].to_dict()
            result = predict_transaction(row_dict)

            if result["prediction"] == "FRAUD":
                st.error(f"🚨 FRAUD — probability: {result['fraud_probability']:.2%}")
            else:
                st.success(f"✅ Legitimate — probability: {result['fraud_probability']:.2%}")

            st.write("### Why the model made this decision (SHAP contributions)")
            st.json(result["top_contributing_features"])

st.divider()
st.caption("Model: XGBoost + SHAP explainability. Drift detection: statistical mean-shift monitoring.")
