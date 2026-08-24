import streamlit as st
import pandas as pd
import requests
import time

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("🔍 Explainable Real-Time Fraud Detection System")
st.caption("Random Forest model + SHAP explainability + drift monitoring")

tab1, tab2 = st.tabs(["Live Feed Simulation", "Manual Transaction Check"])

with tab1:
    st.subheader("Simulated Real-Time Transaction Feed")
    st.write("Streams transactions from a held-out sample one by one to the API.")

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
            st.error("sample_transactions.csv not found in this folder.")
            sample_df = pd.DataFrame()

        results = []
        fraud_flags = []

        for i in range(min(n_transactions, len(sample_df))):
            row = sample_df.iloc[i].to_dict()
            try:
                response = requests.post(f"{API_URL}/predict", json={"features": row}, timeout=5)
                result = response.json()
            except Exception as e:
                st.error(f"Could not reach API: {e}")
                break

            results.append(result)
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

        st.success(f"Simulation complete — processed {len(results)} transactions.")

with tab2:
    st.subheader("Check a Single Transaction Manually")
    uploaded_file = st.file_uploader("Upload a single-transaction CSV", type=["csv"])

    if uploaded_file is not None:
        row_df = pd.read_csv(uploaded_file)
        row_df = row_df.drop(columns=["Class"], errors="ignore")
        st.write("Preview:", row_df.head(1))

        if st.button("Predict this transaction"):
            row_dict = row_df.iloc[0].to_dict()
            try:
                response = requests.post(f"{API_URL}/predict", json={"features": row_dict}, timeout=5)
                result = response.json()

                if result["prediction"] == "FRAUD":
                    st.error(f"🚨 FRAUD — probability: {result['fraud_probability']:.2%}")
                else:
                    st.success(f"✅ Legitimate — probability: {result['fraud_probability']:.2%}")

                st.write("### Why the model made this decision (SHAP contributions)")
                st.json(result["top_contributing_features"])
            except Exception as e:
                st.error(f"Could not reach API: {e}")

st.divider()
st.caption("Backend: FastAPI + Random Forest + SHAP. Dataset: Kaggle Credit Card Fraud Detection.")
