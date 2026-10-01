import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# -----------------------------
# Project paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

PREPARED_DATA_PATH = BASE_DIR / "data" / "processed" / "prepared_credit_data.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = BASE_DIR / "results" / "saved_model"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load prepared dataset
# -----------------------------
data = pd.read_csv(PREPARED_DATA_PATH)
data = data.replace([np.inf, -np.inf], np.nan)

print("Prepared dataset loaded successfully")
print("Prepared dataset shape:", data.shape)

if "TARGET" not in data.columns:
    raise ValueError("TARGET column not found in prepared dataset.")


# -----------------------------
# Define features and target
# -----------------------------
id_columns = [col for col in ["SK_ID_CURR"] if col in data.columns]

X = data.drop(columns=["TARGET"] + id_columns)
y = data["TARGET"]

feature_columns = X.columns.tolist()

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)
print("Number of model features:", len(feature_columns))


# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain/test split complete")
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)


# -----------------------------
# Final model
# -----------------------------
# Class-weighted Logistic Regression is used as the demo model because it best represents
# the responsible AI trade-off: stronger recall, affordability caution, but lower accuracy/precision.
final_model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight="balanced"
    ))
])


# -----------------------------
# Train final model
# -----------------------------
print("\nTraining final model: Class-weighted Logistic Regression")

final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]


# -----------------------------
# Evaluate final model
# -----------------------------
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1-score": f1_score(y_test, y_pred, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, y_proba)
}

print("\nFINAL MODEL PERFORMANCE")
for metric, value in metrics.items():
    print(f"{metric}: {value:.6f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# -----------------------------
# Save model and feature columns
# -----------------------------
model_path = MODEL_DIR / "class_weighted_logistic_regression_model.pkl"
features_path = MODEL_DIR / "final_model_feature_columns.pkl"

joblib.dump(final_model, model_path)
joblib.dump(feature_columns, features_path)

print("\nFinal model saved to:")
print(model_path)

print("\nFeature columns saved to:")
print(features_path)


# -----------------------------
# Save readable model notes
# -----------------------------
notes = """
FINAL SAVED MODEL NOTES

Selected final demo model:
Class-weighted Logistic Regression

Reason for selection:
This model best represents the responsible AI framework because it improves recall for repayment difficulty detection compared with the baseline models. It is not presented as a perfect model. It is used as a prototype model to demonstrate the trade-off between predictive performance, fairness, affordability-risk behaviour and explainability.

Important interpretation:
- TARGET = 0 means no repayment difficulty.
- TARGET = 1 means repayment difficulty.
- A higher predicted probability means higher predicted repayment difficulty risk.
- The model should not be used for real lending decisions.
- The app/demo is for academic prototype demonstration only.

Model input:
The app must provide the same feature columns used during model training. Missing fields in the app will be filled with 0 or default values before prediction.
"""

notes_path = RESULTS_DIR / "saved_model_notes.txt"

with open(notes_path, "w", encoding="utf-8") as file:
    file.write(notes)

print("\nSaved model notes saved to:")
print(notes_path)

print("\nSTEP COMPLETE: Final model trained and saved successfully.")