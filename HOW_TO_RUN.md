# How to Run This Project (Demo Guide)

## Terminal 1 - Start the API

cd ~/fraud-detection-project/api
source ../venv/bin/activate
uvicorn main:app --reload --port 8000

Wait until you see: Application startup complete.
Test it's working: open http://127.0.0.1:8000/docs

## Terminal 2 - Start the Dashboard (open a NEW terminal tab)

cd ~/fraud-detection-project/dashboard
source ../venv/bin/activate
streamlit run app.py

This opens automatically at http://localhost:8501

## What to show in a live demo

1. Live Feed Simulation tab - set transactions to ~50-100, click Start Simulation
2. Manual Transaction Check tab - upload a transaction CSV, see prediction + SHAP explanation
3. API docs page (/docs) - shows backend endpoints if needed

## Shutting down after demo

Press Control+C in each terminal to stop. Safe to close everything - all code is saved in the project folder and pushed to GitHub.

## If something doesn't start

- Make sure you see (venv) in your terminal prompt - if not, run: source ../venv/bin/activate
- Make sure Terminal 1 (API) is running before starting Terminal 2 (dashboard)
- If ports are busy, close old terminal windows from previous sessionsgit add HOW_TO_RUN.md
git commit -m "Add quick-start guide for demos"
git push
