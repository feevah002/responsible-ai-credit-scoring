import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
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

print("FAIRNESS-AWARE IMPROVEMENT: CLASS-WEIGHTED LOGISTIC REGRESSION")
print("--------------------------------------------------------------")

# Load prepared dataset
df = pd.read_csv("data/processed/prepared_credit_data.csv")

print(f"Prepared dataset shape: {df.shape}")

# Separate features and target
X = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
y = df["TARGET"]

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Same train-test split as previous models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Build class-weighted Logistic Regression pipeline
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "class_weighted_logistic_regression",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ]
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

print("\nCLASS-WEIGHTED LOGISTIC REGRESSION PERFORMANCE")
print(metrics)

# Save metrics
metrics.to_csv("results/tables/class_weighted_logistic_regression_metrics.csv", index=False)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

cm_table = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

print("\nCONFUSION MATRIX")
print(cm_table)

cm_table.to_csv("results/tables/class_weighted_logistic_regression_confusion_matrix.csv")

# Save confusion matrix chart
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.title("Class-Weighted Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig("results/charts/class_weighted_logistic_regression_confusion_matrix.png")
plt.close()

# Classification report
report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
report_df = pd.DataFrame(report).transpose()

print("\nCLASSIFICATION REPORT")
print(report_df)

report_df.to_csv("results/tables/class_weighted_logistic_regression_classification_report.csv")

# Save model
joblib.dump(model, "results/model_outputs/class_weighted_logistic_regression_model.pkl")

# Compare with previous models
comparison_files = {
    "Logistic Regression Baseline": "results/tables/logistic_regression_baseline_metrics.csv",
    "Random Forest Baseline": "results/tables/random_forest_baseline_metrics.csv",
    "Class-Weighted Logistic Regression": "results/tables/class_weighted_logistic_regression_metrics.csv"
}

comparison_tables = []

for model_name, file_path in comparison_files.items():
    model_metrics = pd.read_csv(file_path)
    model_metrics = model_metrics.rename(columns={"Value": model_name})
    comparison_tables.append(model_metrics)

comparison = comparison_tables[0]

for table in comparison_tables[1:]:
    comparison = comparison.merge(table, on="Metric")

print("\nTHREE-MODEL PERFORMANCE COMPARISON")
print(comparison)

comparison.to_csv("results/tables/fairness_aware_improvement_model_comparison.csv", index=False)

# Comparison chart
comparison_chart = comparison.set_index("Metric")

comparison_chart.plot(kind="bar", figsize=(10, 6))
plt.title("Baseline Models vs Fairness-Aware Improvement")
plt.ylabel("Score")
plt.xlabel("Metric")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/charts/fairness_aware_improvement_model_comparison.png")
plt.close()

print("\nFAIRNESS-AWARE IMPROVEMENT COMPLETE")
print("-----------------------------------")
print("Metrics saved to: results/tables/class_weighted_logistic_regression_metrics.csv")
print("Confusion matrix saved to: results/tables/class_weighted_logistic_regression_confusion_matrix.csv")
print("Classification report saved to: results/tables/class_weighted_logistic_regression_classification_report.csv")
print("Model saved to: results/model_outputs/class_weighted_logistic_regression_model.pkl")
print("Three-model comparison saved to: results/tables/fairness_aware_improvement_model_comparison.csv")
print("Comparison chart saved to: results/charts/fairness_aware_improvement_model_comparison.png")