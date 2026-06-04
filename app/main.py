from fastapi import FastAPI
import joblib
import pandas as pd

model = joblib.load("models/fraud_model.pkl")

app = FastAPI(                              # API Initialization
    title="Fraud Detection API",
    description="Predicts whether a transaction is fraudulent or legitimate",
    version="1.0"
)


@app.get("/")                               # Home Route
def home():
    return {
        "message": "Fraud Detection API is running!"
    }

@app.post("/predict")                       # Prediction Route
def predict(transaction: dict):

    data = pd.DataFrame([transaction])

    prediction = model.predict(data)[0]

    result = "Fraud" if prediction == 1 else "Legitimate"

    return {
        "prediction": result
    }