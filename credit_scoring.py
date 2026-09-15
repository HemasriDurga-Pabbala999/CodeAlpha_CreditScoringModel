# CodeAlpha - Task 1: Credit Scoring Model
# Machine Learning using Logistic Regression

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)

os.makedirs("results", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset.csv")

X = df.drop("Creditworthy", axis=1)
y = df["Creditworthy"]

numeric_features = [
    "Age", "Annual_Income", "Loan_Amount", "Debt", "Employment_Years"
]
categorical_features = ["Credit_History", "Payment_History"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n===== CREDIT SCORING MODEL RESULTS =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y_test, y_pred,
    target_names=["Not Creditworthy", "Creditworthy"],
    zero_division=0
))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Not Creditworthy", "Creditworthy"]
)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

# ROC curve
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve")
plt.tight_layout()
plt.savefig("results/roc_curve.png")
plt.close()

# Example prediction
sample = pd.DataFrame([{
    "Age": 30,
    "Annual_Income": 65000,
    "Loan_Amount": 12000,
    "Debt": 5000,
    "Credit_History": "Good",
    "Payment_History": "On-Time",
    "Employment_Years": 5
}])

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0, 1]

print("\n===== SAMPLE PREDICTION =====")
print("Prediction:", "Creditworthy" if prediction == 1 else "Not Creditworthy")
print(f"Probability of Creditworthy: {probability:.2%}")
