import joblib
import pandas as pd

model = joblib.load("models/fraud_model.pkl")

print("Model Loaded Successfully!")

df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)

sample_transaction = X.iloc[[0]]

print("\nSample Transaction:")
print(sample_transaction)

prediction = model.predict(sample_transaction)

if prediction[0] == 1:
    print("\n Fraudulent Transaction")
else:
    print("\n Legitimate Transaction")