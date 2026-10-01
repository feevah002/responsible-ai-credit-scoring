import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# -----------------------------
# Project paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

PREPARED_DATA_PATH = BASE_DIR / "data" / "processed" / "prepared_credit_data.csv"

OUTPUT_DIR = BASE_DIR / "results" / "affordability"
CHARTS_DIR = OUTPUT_DIR / "charts"
TABLES_DIR = OUTPUT_DIR / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load prepared dataset
# -----------------------------
data = pd.read_csv(PREPARED_DATA_PATH)
data = data.replace([np.inf, -np.inf], np.nan)

print("Prepared dataset loaded successfully")
print("Prepared dataset shape:", data.shape)

if "TARGET" not in data.columns:
    raise ValueError("TARGET column not found in prepared dataset.")

if "REPAYMENT_BURDEN" not in data.columns:
    raise ValueError("REPAYMENT_BURDEN column not found. Check data preparation script.")


# -----------------------------
# Define features and target
# -----------------------------
id_columns = [col for col in ["SK_ID_CURR"] if col in data.columns]

X = data.drop(columns=["TARGET"] + id_columns)
y = data["TARGET"]

# Keep repayment burden separately for affordability analysis
repayment_burden_all = data["REPAYMENT_BURDEN"]


# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test, rb_train, rb_test = train_test_split(
    X,
    y,
    repayment_burden_all,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain/test split complete")
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)


# -----------------------------
# Repayment burden summary
# -----------------------------
rb_summary = rb_test.describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).reset_index()
rb_summary.columns = ["Metric", "Value"]

rb_summary_path = TABLES_DIR / "repayment_burden_summary.csv"
rb_summary.to_csv(rb_summary_path, index=False)

print("\nREPAYMENT BURDEN SUMMARY")
print(rb_summary)
print(f"Repayment burden summary saved to: {rb_summary_path}")


# High repayment burden threshold
# This is a project proxy, not a regulatory affordability standard.
HIGH_BURDEN_THRESHOLD = 0.35

print("\nHigh repayment burden threshold used:", HIGH_BURDEN_THRESHOLD)


# -----------------------------
# Define models
# -----------------------------
models = {
    "Logistic Regression Baseline": Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42))
    ]),

    "Random Forest Baseline": Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ]),

    "Class-Weighted Logistic Regression": Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight="balanced"
        ))
    ])
}


# -----------------------------
# Helper functions
# -----------------------------
def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, rb_test):
    print(f"\nTraining model: {model_name}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    performance = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    }

    cm = confusion_matrix(y_test, y_pred)

    print("\nPerformance:")
    for key, value in performance.items():
        if key == "Model":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.6f}")

    print("\nConfusion Matrix:")
    print(cm)

    # In this project:
    # TARGET 0 = no repayment difficulty
    # TARGET 1 = repayment difficulty
    # Predicted approval = model predicts 0
    predicted_approval = y_pred == 0
    high_burden = rb_test >= HIGH_BURDEN_THRESHOLD
    approved_high_burden = predicted_approval & high_burden

    total_test = len(y_test)
    total_approved = predicted_approval.sum()
    total_high_burden = high_burden.sum()
    high_burden_approved = approved_high_burden.sum()

    if total_approved > 0:
        percent_approvals_high_burden = (high_burden_approved / total_approved) * 100
        average_rb_approved = rb_test[predicted_approval].mean()
        median_rb_approved = rb_test[predicted_approval].median()
    else:
        percent_approvals_high_burden = np.nan
        average_rb_approved = np.nan
        median_rb_approved = np.nan

    if total_high_burden > 0:
        percent_high_burden_approved = (high_burden_approved / total_high_burden) * 100
    else:
        percent_high_burden_approved = np.nan

    if high_burden_approved > 0:
        actual_difficulty_rate_among_high_burden_approvals = y_test[approved_high_burden].mean() * 100
    else:
        actual_difficulty_rate_among_high_burden_approvals = np.nan

    affordability = {
        "Model": model_name,
        "Test Applicants": total_test,
        "Predicted Approvals": total_approved,
        "Predicted Approval Rate (%)": (total_approved / total_test) * 100,
        "High Burden Threshold": HIGH_BURDEN_THRESHOLD,
        "High Burden Applicants in Test": total_high_burden,
        "High Burden Applicants Approved": high_burden_approved,
        "Approvals with High Burden (%)": percent_approvals_high_burden,
        "High Burden Applicants Approved (%)": percent_high_burden_approved,
        "Average Repayment Burden Among Approvals": average_rb_approved,
        "Median Repayment Burden Among Approvals": median_rb_approved,
        "Actual Difficulty Rate Among High-Burden Approvals (%)": actual_difficulty_rate_among_high_burden_approvals
    }

    # Save top high-burden approved cases for inspection
    case_table = pd.DataFrame({
        "Model": model_name,
        "Actual_TARGET": y_test.values,
        "Predicted_TARGET": y_pred,
        "Predicted_Approval": predicted_approval,
        "Predicted_Risk_Probability": y_proba if y_proba is not None else np.nan,
        "REPAYMENT_BURDEN": rb_test.values
    })

    case_table = case_table[
        (case_table["Predicted_Approval"] == True) &
        (case_table["REPAYMENT_BURDEN"] >= HIGH_BURDEN_THRESHOLD)
    ].sort_values(by="REPAYMENT_BURDEN", ascending=False)

    return performance, affordability, case_table


# -----------------------------
# Run models
# -----------------------------
performance_results = []
affordability_results = []
high_burden_case_tables = []

for model_name, model in models.items():
    performance, affordability, case_table = evaluate_model(
        model_name,
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        rb_test
    )

    performance_results.append(performance)
    affordability_results.append(affordability)
    high_burden_case_tables.append(case_table.head(50))


# -----------------------------
# Save results
# -----------------------------
performance_df = pd.DataFrame(performance_results)
affordability_df = pd.DataFrame(affordability_results)
high_burden_cases_df = pd.concat(high_burden_case_tables, ignore_index=True)

performance_path = TABLES_DIR / "affordability_model_performance.csv"
affordability_path = TABLES_DIR / "model_affordability_risk_summary.csv"
high_burden_cases_path = TABLES_DIR / "high_burden_approved_cases_top50.csv"

performance_df.to_csv(performance_path, index=False)
affordability_df.to_csv(affordability_path, index=False)
high_burden_cases_df.to_csv(high_burden_cases_path, index=False)

print(f"\nModel performance saved to: {performance_path}")
print(f"Affordability risk summary saved to: {affordability_path}")
print(f"High-burden approved cases saved to: {high_burden_cases_path}")

print("\nAFFORDABILITY RISK SUMMARY")
print(affordability_df)


# -----------------------------
# Charts
# -----------------------------
plt.figure(figsize=(10, 6))
plt.bar(
    affordability_df["Model"],
    affordability_df["Approvals with High Burden (%)"]
)
plt.ylabel("Approvals with High Repayment Burden (%)")
plt.title("High Repayment Burden Among Predicted Approvals")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

chart_path = CHARTS_DIR / "high_burden_approvals_by_model.png"
plt.savefig(chart_path, dpi=300)
plt.close()
print(f"Saved chart: {chart_path}")


plt.figure(figsize=(10, 6))
plt.bar(
    affordability_df["Model"],
    affordability_df["Average Repayment Burden Among Approvals"]
)
plt.ylabel("Average Repayment Burden")
plt.title("Average Repayment Burden Among Predicted Approvals")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

chart_path = CHARTS_DIR / "average_repayment_burden_among_approvals.png"
plt.savefig(chart_path, dpi=300)
plt.close()
print(f"Saved chart: {chart_path}")


plt.figure(figsize=(10, 6))
plt.bar(
    affordability_df["Model"],
    affordability_df["Actual Difficulty Rate Among High-Burden Approvals (%)"]
)
plt.ylabel("Actual Difficulty Rate (%)")
plt.title("Actual Repayment Difficulty Among High-Burden Predicted Approvals")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

chart_path = CHARTS_DIR / "difficulty_rate_among_high_burden_approvals.png"
plt.savefig(chart_path, dpi=300)
plt.close()
print(f"Saved chart: {chart_path}")


print("\nSTEP COMPLETE: Affordability-risk analysis finished.")