# Responsible AI Credit Scoring Project

## Project Title

Design and Evaluation of a Fairness-Aware and Affordability-Aware Machine Learning Framework for Credit Scoring

## Project Overview

This project designs and evaluates a responsible AI framework for credit scoring. The aim is not only to predict repayment difficulty, but also to assess model decisions using fairness, affordability-risk, and explainability measures.

The project uses a public borrower-level credit dataset and compares traditional accuracy-focused credit scoring models with a responsible AI evaluation framework.

## Research Question

To what extent can a fairness-aware and affordability-aware machine learning framework improve responsible credit scoring decisions compared with traditional accuracy-focused credit scoring models?

## Main Components

The project includes:

- Data understanding and preparation
- Exploratory data analysis
- Baseline model development
- Model performance comparison
- Fairness audit
- Fairness-aware model improvement
- Affordability-risk analysis
- Explainability and SHAP analysis
- Final responsible AI comparison
- Streamlit dashboard prototype

## Dataset

The project uses the Home Credit application dataset.

The raw dataset was not directly modified. A separate prepared dataset was created for modelling and analysis.

Main processed dataset:

```text
data/processed/prepared_credit_data.csv
```

Fairness reference dataset:

```text
data/processed/fairness_reference.csv
```

## Models Used

The project compares:

1. Logistic Regression baseline
2. Random Forest baseline
3. Class-weighted Logistic Regression

The class-weighted Logistic Regression model was selected for the final prototype because it improved repayment difficulty detection compared with the baseline models.

## Key Results

The baseline Logistic Regression model achieved high accuracy but had very weak recall, meaning it missed most applicants with repayment difficulty.

The Random Forest model improved recall slightly but still remained limited.

The class-weighted Logistic Regression model substantially improved recall, but this came with lower accuracy and precision. It also introduced larger group-level fairness differences.

This supports the project argument that responsible credit scoring should not rely on accuracy alone.

## Responsible AI Evaluation

The project evaluates models using:

- Predictive performance metrics
- Confusion matrices
- Fairness audit metrics
- Affordability-risk indicators
- Feature importance
- Logistic regression coefficients
- SHAP explanations

## Affordability-Risk Analysis

The project uses repayment burden as an affordability-risk proxy:

```text
REPAYMENT_BURDEN = AMT_ANNUITY / AMT_INCOME_TOTAL
```

A repayment burden threshold of 0.35 was used as a project-level high-burden indicator.

This is not a full real-world affordability assessment, but it provides a useful academic proxy for identifying applicants who may need affordability review.

## Streamlit Dashboard

The final dashboard is contained in:

```text
credit_scoring_demo_app.py
```

The dashboard includes:

- Single applicant credit scoring demo
- Model performance comparison
- Fairness audit results
- Affordability-risk analysis
- Explainability and SHAP outputs
- App testing evidence
- Final project summary

To run the dashboard:

```bash
streamlit run credit_scoring_demo_app.py
```

## Project Folders

```text
data/
models/
results/
results/eda/
results/fairness/
results/affordability/
results/explainability/
results/shap/
results/final_comparison/
results/app_testing/
results/dashboard_screenshots/
```

## Important Output Files

Final model:

```text
models/class_weighted_logistic_regression_model.pkl
```

Final model feature columns:

```text
models/final_model_feature_columns.pkl
```

Final comparison outputs:

```text
results/final_comparison/tables/
results/final_comparison/charts/
```

App testing evidence:

```text
results/app_testing/
results/app_testing/screenshots/
```

Dashboard screenshots:

```text
results/dashboard_screenshots/
```

## Limitations

This project is an academic prototype. It is not a production lending system.

Main limitations:

- The dataset is public and historical, not live lender data.
- Repayment burden is only a proxy for affordability.
- The fairness audit is diagnostic and does not prove discrimination.
- The app is a prototype and should not be used for real lending decisions.
- Real-world deployment would require legal compliance, security, monitoring, human review, and a full affordability assessment.

## How to Install Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## How to Run the Project Dashboard

From the project folder, run:

```bash
streamlit run credit_scoring_demo_app.py
```

## Academic Note

This project demonstrates that responsible AI credit scoring is not only about improving prediction accuracy. It also requires fairness auditing, affordability-risk analysis, explainability, and careful interpretation of trade-offs.
