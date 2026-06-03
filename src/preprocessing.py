import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



df = pd.read_csv("data/creditcard.csv")


print("Dataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())


df = df.drop_duplicates()

print("\nShape After Removing Duplicates:", df.shape)


X = df.drop("Class", axis=1)
y = df["Class"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

print("\nFraud Distribution (Train):")
print(y_train.value_counts())

print("\nFraud Distribution (Test):")
print(y_test.value_counts())