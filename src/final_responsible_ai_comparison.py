import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# -----------------------------
# Project paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "results" / "final_comparison"
CHARTS_DIR = OUTPUT_DIR / "charts"
TABLES_DIR = OUTPUT_DIR / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Core model performance results
# -----------------------------
performance_data = [
    {
        "Model": "Logistic Regression Baseline",
        "Accuracy": 0.919450,
        "Precision": 0.569620,
        "Recall": 0.009063,
        "F1-score": 0.017843,
        "ROC-AUC": 0.739034,
        "Main Performance Interpretation": "High accuracy but extremely low recall. Missed most repayment difficulty cases."
    },
    {
        "Model": "Random Forest Baseline",
        "Accuracy": 0.915743,
        "Precision": 0.367844,
        "Recall": 0.060826,
        "F1-score": 0.104390,
        "ROC-AUC": 0.725785,
        "Main Performance Interpretation": "Slightly lower accuracy than Logistic Regression but better recall and F1-score."
    },
    {
        "Model": "Class-Weighted Logistic Regression",
        "Accuracy": 0.685262,
        "Precision": 0.158407,
        "Recall": 0.672105,
        "F1-score": 0.256387,
        "ROC-AUC": 0.739103,
        "Main Performance Interpretation": "Much stronger recall but lower accuracy and precision. Shows a major trade-off."
    }
]

performance_df = pd.DataFrame(performance_data)

performance_path = TABLES_DIR / "final_model_performance_comparison.csv"
performance_df.to_csv(performance_path, index=False)

print("Final model performance comparison saved to:")
print(performance_path)


# -----------------------------
# Fairness summary
# -----------------------------
fairness_data = [
    {
        "Model": "Logistic Regression Baseline",
        "Largest Approval Rate Difference": 0.166667,
        "Largest Approval Difference Variable": "NAME_INCOME_TYPE",
        "Largest False Negative Rate Difference": 0.333333,
        "Fairness Interpretation": "Group-level differences were present. Low recall also limits responsible decision-making."
    },
    {
        "Model": "Random Forest Baseline",
        "Largest Approval Rate Difference": 0.041667,
        "Largest Approval Difference Variable": "OCCUPATION_TYPE",
        "Largest False Negative Rate Difference": 0.125000,
        "Fairness Interpretation": "Approval rate differences were smaller than Logistic Regression, but false negative differences remained."
    },
    {
        "Model": "Class-Weighted Logistic Regression",
        "Largest Approval Rate Difference": 1.000000,
        "Largest Approval Difference Variable": "NAME_INCOME_TYPE",
        "Largest False Negative Rate Difference": 1.000000,
        "Fairness Interpretation": "Recall improved strongly, but some group-level fairness differences increased. This shows a fairness-performance trade-off."
    }
]

fairness_df = pd.DataFrame(fairness_data)

fairness_path = TABLES_DIR / "final_fairness_tradeoff_summary.csv"
fairness_df.to_csv(fairness_path, index=False)

print("\nFinal fairness trade-off summary saved to:")
print(fairness_path)


# -----------------------------
# Affordability-risk summary
# -----------------------------
affordability_csv = BASE_DIR / "results" / "affordability" / "tables" / "model_affordability_risk_summary.csv"

if affordability_csv.exists():
    affordability_df = pd.read_csv(affordability_csv)
    print("\nLoaded affordability-risk summary from existing file.")
else:
    print("\nAffordability file not found. Using known project values.")

    affordability_df = pd.DataFrame([
        {
            "Model": "Logistic Regression Baseline",
            "Predicted Approvals": 61423,
            "Predicted Approval Rate (%)": 99.87,
            "High Burden Threshold": 0.35,
            "High Burden Applicants in Test": 3258,
            "High Burden Applicants Approved": 3252,
            "High Burden Applicants Approved (%)": 99.82,
            "Actual Difficulty Rate Among High-Burden Approvals (%)": 7.656827
        },
        {
            "Model": "Random Forest Baseline",
            "Predicted Approvals": 60681,
            "Predicted Approval Rate (%)": 98.67,
            "High Burden Threshold": 0.35,
            "High Burden Applicants in Test": 3258,
            "High Burden Applicants Approved": 3234,
            "High Burden Applicants Approved (%)": 99.26,
            "Actual Difficulty Rate Among High-Burden Approvals (%)": 7.452072
        },
        {
            "Model": "Class-Weighted Logistic Regression",
            "Predicted Approvals": 40436,
            "Predicted Approval Rate (%)": 65.75,
            "High Burden Threshold": 0.35,
            "High Burden Applicants in Test": 3258,
            "High Burden Applicants Approved": 1922,
            "High Burden Applicants Approved (%)": 58.99,
            "Actual Difficulty Rate Among High-Burden Approvals (%)": 3.798127
        }
    ])

affordability_path = TABLES_DIR / "final_affordability_risk_summary.csv"
affordability_df.to_csv(affordability_path, index=False)

print("Final affordability-risk summary saved to:")
print(affordability_path)


# -----------------------------
# Explainability and SHAP summary
# -----------------------------
shap_csv = BASE_DIR / "results" / "shap" / "tables" / "combined_top_shap_features.csv"
explainability_csv = BASE_DIR / "results" / "explainability" / "tables" / "combined_top_explainability_features.csv"

explainability_summary_rows = []

if shap_csv.exists():
    shap_df = pd.read_csv(shap_csv)

    for model_name in shap_df["Model"].unique():
        model_features = shap_df[shap_df["Model"] == model_name]["Feature"].head(10).tolist()

        explainability_summary_rows.append({
            "Model": model_name,
            "Explainability Method": "SHAP",
            "Top Features": ", ".join(model_features),
            "Interpretation": "SHAP identifies features with the strongest average contribution to repayment difficulty predictions."
        })

if explainability_csv.exists():
    expl_df = pd.read_csv(explainability_csv)

    for model_name in expl_df["Model"].unique():
        model_features = expl_df[expl_df["Model"] == model_name]["Feature"].head(10).tolist()

        explainability_summary_rows.append({
            "Model": model_name,
            "Explainability Method": "Feature importance / coefficients",
            "Top Features": ", ".join(model_features),
            "Interpretation": "Global explainability confirms which variables had strongest influence in the model."
        })

explainability_summary_df = pd.DataFrame(explainability_summary_rows)

explainability_summary_path = TABLES_DIR / "final_explainability_summary.csv"
explainability_summary_df.to_csv(explainability_summary_path, index=False)

print("\nFinal explainability summary saved to:")
print(explainability_summary_path)


# -----------------------------
# Responsible AI final comparison table
# -----------------------------
final_summary_data = [
    {
        "Model": "Logistic Regression Baseline",
        "Predictive Performance": "High accuracy but very weak recall.",
        "Fairness": "Group-level differences identified, especially by income type.",
        "Affordability Risk": "Approved almost all high-burden applicants.",
        "Explainability": "Coefficients and SHAP show influence of credit amount, external scores, occupation, gender and repayment burden.",
        "Overall Responsible AI Assessment": "Strong accuracy-focused baseline but weak responsible AI performance because it misses most repayment difficulty cases."
    },
    {
        "Model": "Random Forest Baseline",
        "Predictive Performance": "Slightly lower accuracy but better recall than Logistic Regression.",
        "Fairness": "Smaller approval-rate difference than Logistic Regression but still has group-level differences.",
        "Affordability Risk": "Also approved almost all high-burden applicants.",
        "Explainability": "Feature importance and SHAP show influence of external scores, age, employment, annuity and repayment burden.",
        "Overall Responsible AI Assessment": "Better minority-class detection than Logistic Regression, but still limited as a responsible credit scoring model."
    },
    {
        "Model": "Class-Weighted Logistic Regression",
        "Predictive Performance": "Much higher recall but lower accuracy and precision.",
        "Fairness": "Improved repayment difficulty detection but increased some group-level fairness differences.",
        "Affordability Risk": "More cautious with high-burden applicants and lower actual difficulty rate among high-burden approvals.",
        "Explainability": "Coefficients and SHAP show repayment burden contributes to repayment difficulty prediction.",
        "Overall Responsible AI Assessment": "Most useful responsible AI comparison model because it exposes trade-offs between recall, fairness, affordability and accuracy."
    }
]

final_summary_df = pd.DataFrame(final_summary_data)

final_summary_path = TABLES_DIR / "final_responsible_ai_comparison_table.csv"
final_summary_df.to_csv(final_summary_path, index=False)

print("\nFinal responsible AI comparison table saved to:")
print(final_summary_path)


# -----------------------------
# Research question answer note
# -----------------------------
research_question_answer = """
FINAL RESPONSIBLE AI COMPARISON NOTES

Research question:
To what extent can a fairness-aware and affordability-aware machine learning framework improve responsible credit scoring decisions compared with traditional accuracy-focused credit scoring models?

Summary answer:
The responsible AI framework improves the evaluation of credit scoring decisions by showing limitations that accuracy alone does not reveal. The Logistic Regression baseline achieved high accuracy, but its recall was extremely low, meaning it missed most applicants with repayment difficulty. Random Forest improved recall slightly but still remained limited.

The class-weighted Logistic Regression model substantially improved recall, meaning it identified many more applicants with repayment difficulty. However, this came with lower accuracy and precision, and it increased some group-level fairness differences. Therefore, the fairness-aware improvement did not simply solve the problem. Instead, it exposed an important trade-off between detecting repayment difficulty and maintaining stable group-level fairness outcomes.

The affordability-risk analysis added another layer of evaluation. Baseline models approved almost all high-burden applicants, while the class-weighted model was more cautious and approved fewer high-burden applicants. The actual difficulty rate among high-burden predicted approvals was also lower for the class-weighted model. However, repayment burden is only a proxy and does not represent a full affordability assessment.

The explainability and SHAP analysis showed that model predictions were influenced by external source scores, credit amount variables, borrower profile variables, and repayment burden. The appearance of REPAYMENT_BURDEN among important features supports the affordability-aware part of the framework.

Overall, the project shows that a responsible AI framework does not necessarily produce a single perfect model. Its main value is that it provides a wider evaluation of credit scoring by comparing predictive performance, fairness, affordability risk and explainability together.
"""

rq_path = TABLES_DIR / "final_research_question_answer_notes.txt"

with open(rq_path, "w", encoding="utf-8") as file:
    file.write(research_question_answer)

print("\nFinal research question answer notes saved to:")
print(rq_path)


# -----------------------------
# Charts
# -----------------------------
plt.figure(figsize=(10, 6))
plt.bar(performance_df["Model"], performance_df["Recall"])
plt.ylabel("Recall")
plt.title("Recall Comparison Across Models")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

recall_chart_path = CHARTS_DIR / "final_recall_comparison.png"
plt.savefig(recall_chart_path, dpi=300)
plt.close()

print("\nSaved chart:")
print(recall_chart_path)


plt.figure(figsize=(10, 6))
plt.bar(performance_df["Model"], performance_df["Accuracy"])
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison Across Models")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

accuracy_chart_path = CHARTS_DIR / "final_accuracy_comparison.png"
plt.savefig(accuracy_chart_path, dpi=300)
plt.close()

print("Saved chart:")
print(accuracy_chart_path)


# Chart for actual difficulty among high-burden approvals
difficulty_column = "Actual Difficulty Rate Among High-Burden Approvals (%)"

if difficulty_column in affordability_df.columns:
    plt.figure(figsize=(10, 6))
    plt.bar(affordability_df["Model"], affordability_df[difficulty_column])
    plt.ylabel("Actual Difficulty Rate (%)")
    plt.title("Actual Difficulty Rate Among High-Burden Predicted Approvals")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    difficulty_chart_path = CHARTS_DIR / "final_high_burden_difficulty_comparison.png"
    plt.savefig(difficulty_chart_path, dpi=300)
    plt.close()

    print("Saved chart:")
    print(difficulty_chart_path)


# Chart for fairness approval-rate difference
plt.figure(figsize=(10, 6))
plt.bar(fairness_df["Model"], fairness_df["Largest Approval Rate Difference"])
plt.ylabel("Largest Approval Rate Difference")
plt.title("Largest Approval Rate Difference Across Models")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

fairness_chart_path = CHARTS_DIR / "final_fairness_approval_difference_comparison.png"
plt.savefig(fairness_chart_path, dpi=300)
plt.close()

print("Saved chart:")
print(fairness_chart_path)


print("\nSTEP COMPLETE: Final responsible AI comparison finished.")