from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Fraud Detection API")

# load trained artifacts
model = joblib.load("xgb_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def root():
    return {"status": "Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: dict):
    df = pd.DataFrame([transaction])

    # scale numeric columns
    df[['Amount', 'Time']] = scaler.transform(df[['Amount', 'Time']])

    prob = model.predict_proba(df)[0][1]

    if prob < 0.3:
        decision = "LOW RISK – Allow"
    elif prob < 0.7:
        decision = "MEDIUM RISK – Review"
    else:
        decision = "HIGH RISK – Block"

    return {
        "fraud_probability": round(float(prob), 4),
        "decision": decision
    }
