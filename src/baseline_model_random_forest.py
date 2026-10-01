import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Create output folders
BASE_DIR / "results" / "tables".mkdir(parents=True, exist_ok=True)
Path("results/charts").mkdir(parents=True, exist_ok=True)
Path("results/model_outputs").mkdir(parents=True, exist_ok=True)

print("BASELINE MODEL: RANDOM FOREST")
print("-----------------------------")

# Load prepared dataset
df = pd.read_csv("data/processed/prepared_credit_data.csv")

print(f"Prepared dataset shape: {df.shape}")

# Separate features and target
X = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
y = df["TARGET"]

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Build Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)

metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
    "Value": [accuracy, precision, recall, f1, roc_auc]
})

print("\nRANDOM FOREST MODEL PERFORMANCE")
print(metrics)

# Save metrics
metrics.to_csv("results/tables/random_forest_baseline_metrics.csv", index=False)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

cm_table = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

print("\nCONFUSION MATRIX")
print(cm_table)

cm_table.to_csv("results/tables/random_forest_confusion_matrix.csv")

# Save confusion matrix chart
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("results/charts/random_forest_confusion_matrix.png")
plt.close()

# Classification report
report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("results/tables/random_forest_classification_report.csv")

print("\nCLASSIFICATION REPORT")
print(report_df)

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

feature_importance.to_csv("results/tables/random_forest_feature_importance.csv", index=False)

print("\nTOP 15 IMPORTANT FEATURES")
print(feature_importance.head(15))

# Save feature importance chart
top_features = feature_importance.head(15)

plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Random Forest Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("results/charts/random_forest_feature_importance.png")
plt.close()

# Save model
joblib.dump(model, "results/model_outputs/random_forest_baseline_model.pkl")

print("\nRANDOM FOREST MODEL COMPLETE")
print("----------------------------")
print("Metrics saved to: results/tables/random_forest_baseline_metrics.csv")
print("Confusion matrix saved to: results/tables/random_forest_confusion_matrix.csv")
print("Classification report saved to: results/tables/random_forest_classification_report.csv")
print("Feature importance saved to: results/tables/random_forest_feature_importance.csv")
print("Confusion matrix chart saved to: results/charts/random_forest_confusion_matrix.png")
print("Feature importance chart saved to: results/charts/random_forest_feature_importance.png")
print("Model saved to: results/model_outputs/random_forest_baseline_model.pkl")