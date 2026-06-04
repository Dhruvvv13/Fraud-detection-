import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/creditcard.csv")

df = df.drop_duplicates()

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

print("Training Final Model...")

model.fit(X_train, y_train)

print("Training Complete!")

joblib.dump(model, "models/fraud_model.pkl")

print("\nModel Saved Successfully!")
print("Location: models/fraud_model.pkl")