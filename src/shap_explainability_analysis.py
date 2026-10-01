import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import shap

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# Project paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

PREPARED_DATA_PATH = BASE_DIR / "data" / "processed" / "prepared_credit_data.csv"

OUTPUT_DIR = BASE_DIR / "results" / "shap"
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


# -----------------------------
# Define features and target
# -----------------------------
id_columns = [col for col in ["SK_ID_CURR"] if col in data.columns]

X = data.drop(columns=["TARGET"] + id_columns)
y = data["TARGET"]

feature_names = X.columns.tolist()

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)


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
# Sampling for SHAP
# -----------------------------
# SHAP can be slow on 61,502 test rows, so we use a representative sample.
SHAP_SAMPLE_SIZE = 300
BACKGROUND_SAMPLE_SIZE = 300

X_test_sample = X_test.sample(
    n=min(SHAP_SAMPLE_SIZE, len(X_test)),
    random_state=42
)

X_train_background = X_train.sample(
    n=min(BACKGROUND_SAMPLE_SIZE, len(X_train)),
    random_state=42
)

print("\nSHAP sample created")
print("SHAP test sample shape:", X_test_sample.shape)
print("SHAP background sample shape:", X_train_background.shape)


# -----------------------------
# Helper: extract class 1 SHAP values safely
# -----------------------------
def extract_class_1_shap_values(shap_output):
    """
    Handles different SHAP output formats.
    For binary classification, we want class 1 because TARGET = 1 means repayment difficulty.
    """
    if isinstance(shap_output, list):
        if len(shap_output) > 1:
            return shap_output[1]
        return shap_output[0]

    values = shap_output.values if hasattr(shap_output, "values") else shap_output

    # Possible shape: samples x features x classes
    if len(values.shape) == 3:
        if values.shape[2] == 2:
            return values[:, :, 1]
        if values.shape[1] == 2:
            return values[:, 1, :]

    return values


# -----------------------------
# Helper: save SHAP outputs
# -----------------------------
def save_shap_outputs(model_name, shap_values, X_sample, safe_name):
    shap_values_class_1 = extract_class_1_shap_values(shap_values)

    mean_abs_shap = np.abs(shap_values_class_1).mean(axis=0)

    shap_importance_df = pd.DataFrame({
        "Feature": X_sample.columns,
        "Mean_Absolute_SHAP_Value": mean_abs_shap
    }).sort_values(
        by="Mean_Absolute_SHAP_Value",
        ascending=False
    )

    table_path = TABLES_DIR / f"{safe_name}_mean_absolute_shap_values.csv"
    shap_importance_df.to_csv(table_path, index=False)

    print(f"{model_name} SHAP importance saved to: {table_path}")

    # Matplotlib bar chart
    top_15 = shap_importance_df.head(15).sort_values(
        by="Mean_Absolute_SHAP_Value",
        ascending=True
    )

    plt.figure(figsize=(10, 7))
    plt.barh(top_15["Feature"], top_15["Mean_Absolute_SHAP_Value"])
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.title(f"Top SHAP Features: {model_name}")
    plt.tight_layout()

    chart_path = CHARTS_DIR / f"{safe_name}_top_shap_features.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()

    print(f"{model_name} SHAP bar chart saved to: {chart_path}")

    # SHAP summary plot
    plt.figure()
    shap.summary_plot(
        shap_values_class_1,
        X_sample,
        show=False,
        max_display=15
    )
    plt.tight_layout()

    summary_path = CHARTS_DIR / f"{safe_name}_shap_summary_plot.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"{model_name} SHAP summary plot saved to: {summary_path}")

    return shap_importance_df


# -----------------------------
# Model 1: Logistic Regression Baseline
# -----------------------------
print("\nTraining Logistic Regression Baseline...")

lr_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000, random_state=42))
])

lr_pipeline.fit(X_train, y_train)

lr_imputer = lr_pipeline.named_steps["imputer"]
lr_scaler = lr_pipeline.named_steps["scaler"]
lr_model = lr_pipeline.named_steps["model"]

X_train_lr_bg = pd.DataFrame(
    lr_scaler.transform(lr_imputer.transform(X_train_background)),
    columns=feature_names
)

X_test_lr_sample = pd.DataFrame(
    lr_scaler.transform(lr_imputer.transform(X_test_sample)),
    columns=feature_names
)

print("Generating SHAP values for Logistic Regression Baseline...")

lr_explainer = shap.LinearExplainer(
    lr_model,
    X_train_lr_bg
)

lr_shap_values = lr_explainer(X_test_lr_sample)

lr_shap_df = save_shap_outputs(
    model_name="Logistic Regression Baseline",
    shap_values=lr_shap_values,
    X_sample=X_test_lr_sample,
    safe_name="logistic_regression_baseline"
)


# -----------------------------
# Model 2: Random Forest Baseline
# -----------------------------
print("\nTraining Random Forest Baseline...")

rf_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

rf_pipeline.fit(X_train, y_train)

rf_imputer = rf_pipeline.named_steps["imputer"]
rf_model = rf_pipeline.named_steps["model"]

X_train_rf_bg = pd.DataFrame(
    rf_imputer.transform(X_train_background),
    columns=feature_names
)

X_test_rf_sample = pd.DataFrame(
    rf_imputer.transform(X_test_sample),
    columns=feature_names
)

print("Generating SHAP values for Random Forest Baseline...")

rf_explainer = shap.TreeExplainer(rf_model)

rf_shap_values = rf_explainer(X_test_rf_sample)

rf_shap_df = save_shap_outputs(
    model_name="Random Forest Baseline",
    shap_values=rf_shap_values,
    X_sample=X_test_rf_sample,
    safe_name="random_forest_baseline"
)


# -----------------------------
# Model 3: Class-Weighted Logistic Regression
# -----------------------------
print("\nTraining Class-Weighted Logistic Regression...")

cw_lr_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight="balanced"
    ))
])

cw_lr_pipeline.fit(X_train, y_train)

cw_lr_imputer = cw_lr_pipeline.named_steps["imputer"]
cw_lr_scaler = cw_lr_pipeline.named_steps["scaler"]
cw_lr_model = cw_lr_pipeline.named_steps["model"]

X_train_cw_lr_bg = pd.DataFrame(
    cw_lr_scaler.transform(cw_lr_imputer.transform(X_train_background)),
    columns=feature_names
)

X_test_cw_lr_sample = pd.DataFrame(
    cw_lr_scaler.transform(cw_lr_imputer.transform(X_test_sample)),
    columns=feature_names
)

print("Generating SHAP values for Class-Weighted Logistic Regression...")

cw_lr_explainer = shap.LinearExplainer(
    cw_lr_model,
    X_train_cw_lr_bg
)

cw_lr_shap_values = cw_lr_explainer(X_test_cw_lr_sample)

cw_lr_shap_df = save_shap_outputs(
    model_name="Class-Weighted Logistic Regression",
    shap_values=cw_lr_shap_values,
    X_sample=X_test_cw_lr_sample,
    safe_name="class_weighted_logistic_regression"
)


# -----------------------------
# Combined SHAP top features
# -----------------------------
combined_rows = []

for model_name, shap_df in [
    ("Logistic Regression Baseline", lr_shap_df),
    ("Random Forest Baseline", rf_shap_df),
    ("Class-Weighted Logistic Regression", cw_lr_shap_df)
]:
    for _, row in shap_df.head(10).iterrows():
        combined_rows.append({
            "Model": model_name,
            "Feature": row["Feature"],
            "Mean_Absolute_SHAP_Value": row["Mean_Absolute_SHAP_Value"]
        })

combined_shap_df = pd.DataFrame(combined_rows)

combined_path = TABLES_DIR / "combined_top_shap_features.csv"
combined_shap_df.to_csv(combined_path, index=False)

print(f"\nCombined top SHAP features saved to: {combined_path}")


# -----------------------------
# SHAP notes
# -----------------------------
notes = """
SHAP EXPLAINABILITY ANALYSIS NOTES

This stage adds SHAP explainability to the responsible AI credit scoring framework.

Purpose:
- To identify which features contributed most strongly to model predictions.
- To support explainability beyond accuracy, fairness and affordability-risk analysis.
- To compare whether the baseline and class-weighted models rely on similar or different features.

Interpretation:
- Mean absolute SHAP values show the average strength of a feature's contribution to predictions.
- Larger mean absolute SHAP values indicate stronger overall influence.
- For this project, class 1 is the important class because TARGET = 1 means repayment difficulty.
- SHAP values should be interpreted as model explanation, not causal evidence.

Models explained:
- Logistic Regression Baseline using LinearExplainer.
- Random Forest Baseline using TreeExplainer.
- Class-Weighted Logistic Regression using LinearExplainer.

Limitations:
- SHAP was calculated on a representative test sample to reduce computation time.
- The analysis provides explanation of model behaviour, not proof that variables cause repayment difficulty.
- For linear models, SHAP values are based on transformed/scaled features.
"""

notes_path = TABLES_DIR / "shap_explainability_notes.txt"

with open(notes_path, "w", encoding="utf-8") as file:
    file.write(notes)

print(f"SHAP notes saved to: {notes_path}")

print("\nSTEP COMPLETE: SHAP explainability analysis finished.")