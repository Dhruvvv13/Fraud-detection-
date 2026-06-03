import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE

df = pd.read_csv("data/creditcard.csv")

df = df.drop_duplicates()

X = df.drop("Class", axis=1)                            # Features (X = Everything except Class)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(    # Train-Test Split
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()                              # Initialize scaler

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

smote = SMOTE(random_state=42)                         # Initialize SMOTE

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)
     
print("Before SMOTE:")                                 
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_smote, y_train_smote)

y_pred = model.predict(X_test)                         # PREDICTIONS

print("\nAccuracy:")                                   # EVALUATION
print(accuracy_score(y_test, y_pred))

print("\nPrecision:")
print(precision_score(y_test, y_pred))

print("\nRecall:")
print(recall_score(y_test, y_pred))

print("\nF1 Score:")
print(f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))