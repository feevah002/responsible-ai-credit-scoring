import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
import joblib

from sklearn.model_selection import train_test_split

# Create output folders
BASE_DIR / "results" / "tables".mkdir(parents=True, exist_ok=True)
Path("results/charts").mkdir(parents=True, exist_ok=True)

print("FAIRNESS AUDIT WITH IMPROVED MODEL")
print("----------------------------------")

# Load datasets
prepared_df = pd.read_csv("data/processed/prepared_credit_data.csv")
fairness_reference = pd.read_csv("data/processed/fairness_reference.csv")

print(f"Prepared dataset shape: {prepared_df.shape}")
print(f"Fairness reference shape: {fairness_reference.shape}")

# Features, target and IDs
X = prepared_df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
y = prepared_df["TARGET"]
ids = prepared_df["SK_ID_CURR"]

# Same train-test split as previous scripts
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X,
    y,
    ids,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load trained models
models = {
    "Logistic Regression": joblib.load("results/model_outputs/logistic_regression_baseline_model.pkl"),
    "Random Forest": joblib.load("results/model_outputs/random_forest_baseline_model.pkl"),
    "Class-Weighted Logistic Regression": joblib.load("results/model_outputs/class_weighted_logistic_regression_model.pkl")
}

# Group variables for fairness audit
group_columns = [
    "CODE_GENDER",
    "AGE_GROUP",
    "INCOME_GROUP",
    "NAME_FAMILY_STATUS",
    "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE"
]

group_columns = [col for col in group_columns if col in fairness_reference.columns]

fairness_reference = fairness_reference[["SK_ID_CURR"] + group_columns].copy()


def create_prediction_dataframe(model, model_name):
    y_pred = model.predict(X_test)

    predictions = pd.DataFrame({
        "SK_ID_CURR": ids_test.reset_index(drop=True),
        "TARGET": y_test.reset_index(drop=True),
        "PREDICTED_RISK": y_pred
    })

    # TARGET = 1 means repayment difficulty.
    # Predicted approval means predicted no repayment difficulty.
    predictions["PREDICTED_APPROVAL"] = (predictions["PREDICTED_RISK"] == 0).astype(int)
    predictions["MODEL"] = model_name

    return predictions


def calculate_group_metrics(data, group_column, model_name):
    rows = []

    data[group_column] = data[group_column].astype("object").fillna("Unknown")

    for group_value, group_data in data.groupby(group_column):
        total = len(group_data)

        actual_no_difficulty = group_data[group_data["TARGET"] == 0]
        actual_difficulty = group_data[group_data["TARGET"] == 1]

        approval_rate = group_data["PREDICTED_APPROVAL"].mean()
        predicted_risk_rate = group_data["PREDICTED_RISK"].mean()
        actual_risk_rate = group_data["TARGET"].mean()

        if len(actual_no_difficulty) > 0:
            false_positive_rate = (actual_no_difficulty["PREDICTED_RISK"] == 1).mean()
        else:
            false_positive_rate = np.nan

        if len(actual_difficulty) > 0:
            false_negative_rate = (actual_difficulty["PREDICTED_RISK"] == 0).mean()
        else:
            false_negative_rate = np.nan

        rows.append({
            "Model": model_name,
            "Group Variable": group_column,
            "Group": group_value,
            "Count": total,
            "Actual Risk Rate": actual_risk_rate,
            "Approval Rate": approval_rate,
            "Predicted Risk Rate": predicted_risk_rate,
            "False Positive Rate": false_positive_rate,
            "False Negative Rate": false_negative_rate
        })

    return pd.DataFrame(rows)


def create_difference_summary(group_results):
    summary_rows = []

    for (model_name, group_variable), subset in group_results.groupby(["Model", "Group Variable"]):
        summary_rows.append({
            "Model": model_name,
            "Group Variable": group_variable,
            "Approval Rate Difference": subset["Approval Rate"].max() - subset["Approval Rate"].min(),
            "Predicted Risk Rate Difference": subset["Predicted Risk Rate"].max() - subset["Predicted Risk Rate"].min(),
            "False Positive Rate Difference": subset["False Positive Rate"].max() - subset["False Positive Rate"].min(),
            "False Negative Rate Difference": subset["False Negative Rate"].max() - subset["False Negative Rate"].min()
        })

    return pd.DataFrame(summary_rows)


all_group_results = []

for model_name, model in models.items():
    predictions = create_prediction_dataframe(model, model_name)
    fairness_data = predictions.merge(fairness_reference, on="SK_ID_CURR", how="left")

    for group_column in group_columns:
        group_metrics = calculate_group_metrics(fairness_data, group_column, model_name)
        all_group_results.append(group_metrics)

# Combine all results
fairness_results = pd.concat(all_group_results, ignore_index=True)

# Save full group-level fairness results
fairness_results.to_csv(
    "results/tables/fairness_audit_with_improved_model_group_results.csv",
    index=False
)

# Create and save difference summary
fairness_summary = create_difference_summary(fairness_results)

fairness_summary.to_csv(
    "results/tables/fairness_audit_with_improved_model_difference_summary.csv",
    index=False
)

# Print full summary clearly
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)

print("\nFAIRNESS DIFFERENCE SUMMARY")
print(fairness_summary.to_string(index=False))

# Sort by approval rate difference
approval_sorted = fairness_summary.sort_values(
    by="Approval Rate Difference",
    ascending=False
)

print("\nTOP FAIRNESS DIFFERENCES BY APPROVAL RATE")
print(approval_sorted.to_string(index=False))

approval_sorted.to_csv(
    "results/tables/fairness_audit_with_improved_model_sorted_by_approval_difference.csv",
    index=False
)

# Sort by false negative rate difference
fnr_sorted = fairness_summary.sort_values(
    by="False Negative Rate Difference",
    ascending=False
)

print("\nTOP FAIRNESS DIFFERENCES BY FALSE NEGATIVE RATE")
print(fnr_sorted.to_string(index=False))

fnr_sorted.to_csv(
    "results/tables/fairness_audit_with_improved_model_sorted_by_false_negative_difference.csv",
    index=False
)

# Create approval rate charts for each model and group variable
for model_name in fairness_results["Model"].unique():
    for group_column in group_columns:
        chart_data = fairness_results[
            (fairness_results["Model"] == model_name) &
            (fairness_results["Group Variable"] == group_column)
        ].copy()

        chart_data = chart_data.sort_values("Count", ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        plt.bar(chart_data["Group"].astype(str), chart_data["Approval Rate"])
        plt.title(f"{model_name} Approval Rate by {group_column}")
        plt.xlabel(group_column)
        plt.ylabel("Approval Rate")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        safe_model_name = model_name.lower().replace(" ", "_").replace("-", "_")
        safe_group_name = group_column.lower()

        plt.savefig(
            f"results/charts/{safe_model_name}_approval_rate_by_{safe_group_name}.png"
        )
        plt.close()

print("\nFAIRNESS AUDIT WITH IMPROVED MODEL COMPLETE")
print("------------------------------------------")
print("Full group results saved to: results/tables/fairness_audit_with_improved_model_group_results.csv")
print("Difference summary saved to: results/tables/fairness_audit_with_improved_model_difference_summary.csv")
print("Sorted approval difference saved to: results/tables/fairness_audit_with_improved_model_sorted_by_approval_difference.csv")
print("Sorted false negative difference saved to: results/tables/fairness_audit_with_improved_model_sorted_by_false_negative_difference.csv")
print("Approval rate charts saved to: results/charts")