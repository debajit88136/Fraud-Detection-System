
## Running Locally

### Option 1: Combined version (simplest)

```bash
git clone https://github.com/debajit88136/Fraud-Detection-System.git
cd Fraud-Detection-System
python3 -m venv venv
source venv/bin/activate
pip install -r deploy/requirements.txt
cd deploy
streamlit run app.py
```

### Option 2: Microservice version (API + Dashboard separately)

```bash
pip install pandas numpy scikit-learn xgboost shap joblib fastapi uvicorn streamlit requests pydantic

# Terminal 1
cd api
uvicorn main:app --reload --port 8000

# Terminal 2
cd dashboard
streamlit run app.py
```

See `HOW_TO_RUN.md` for more details.

## Evaluation Metrics - Why Not Accuracy

With fraud making up 0.17% of transactions, accuracy is misleading by design. This project reports Precision, Recall, F1-score, and ROC-AUC instead, since these properly capture performance on the rare (fraud) class.

## Possible Extensions

- Replace the custom drift detector with a production-grade library (e.g. evidently)
- Persist prediction logs to a database instead of in-memory storage
- Add an automated retraining trigger when drift is detected repeatedly
- Add API authentication
- Separate deployment of the FastAPI backend (e.g. on Render) for a true microservice architecture in production

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) — anonymized real-world credit card transactions, made available by the Machine Learning Group at ULB on Kaggle.
