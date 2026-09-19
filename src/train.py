import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import create_preprocessor


df = pd.read_csv("data/customer_churn.csv")

## cleaning data
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna(
    subset=["TotalCharges"]
)

df = df.drop(
    columns=["customerID"]
)
## X = Features , y = Target

X = df.drop(
    columns=["Churn"]
)

y = df["Churn"]

## Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

## Create Preprocessor (دلوقتي نستدعي الـfunction اللي عملناها:)
preprocessor = create_preprocessor()

## Create ML Pipeline
model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])
# Train
model.fit(
    X_train,
    y_train
)  
# Prediction
y_pred = model.predict(
    X_test
)
# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    "Accuracy:",
    accuracy
)
# Classification Report
print(
    classification_report(
        y_test,
        y_pred
    )
)
# Save Model
joblib.dump(
    model,
    "models/churn_model.pkl"
)
print("Model saved successfully!")

