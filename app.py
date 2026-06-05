import gradio as gr
import joblib
import pandas as pd

model = joblib.load("models/fraud_model.pkl")

def predict_transaction(*features):            # Prediction Function

    columns = [
        "Time","V1","V2","V3","V4","V5","V6","V7","V8",
        "V9","V10","V11","V12","V13","V14","V15","V16",
        "V17","V18","V19","V20","V21","V22","V23","V24",
        "V25","V26","V27","V28","Amount"
    ]

    data = pd.DataFrame([features], columns=columns)

    prediction = model.predict(data)[0]

    if prediction == 1:
        return "🚨 Fraudulent Transaction"
    else:
        return "✅ Legitimate Transaction"


inputs = []                           # Create Input Fields for Each Feature

feature_names = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8",
    "V9","V10","V11","V12","V13","V14","V15","V16",
    "V17","V18","V19","V20","V21","V22","V23","V24",
    "V25","V26","V27","V28","Amount"
]

for feature in feature_names:
    inputs.append(
        gr.Number(label=feature, value=0)
    )

app = gr.Interface(
    fn=predict_transaction,
    inputs=inputs,
    outputs="text",
    title="Fraud Detection System",
    description="Enter transaction details and predict whether the transaction is fraudulent."
)

app.launch()