import pandas as pd
import numpy as np
from pathlib import Path

# Create output folders
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("results/tables").mkdir(parents=True, exist_ok=True)

# Load raw dataset
df = pd.read_csv("data/raw/home_credit/application_train.csv")

print("DATA PREPARATION STARTED")
print("------------------------")
print(f"Original dataset shape: {df.shape}")

# Keep a copy of the original row count
original_rows = len(df)

# Select useful columns for this project
selected_columns = [
    "SK_ID_CURR",
    "TARGET",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "REG_CITY_NOT_WORK_CITY",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3"
]

# Keep only columns that exist in the dataset
selected_columns = [col for col in selected_columns if col in df.columns]
df = df[selected_columns].copy()

print(f"Selected dataset shape: {df.shape}")

# Remove unclear gender category if present
if "CODE_GENDER" in df.columns:
    df = df[df["CODE_GENDER"] != "XNA"].copy()

# Handle unusual DAYS_EMPLOYED value used in the dataset
# 365243 is treated as an unknown/anomalous value
if "DAYS_EMPLOYED" in df.columns:
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

# Create age in years
df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).round(1)

# Create employment duration in years
df["EMPLOYMENT_YEARS"] = (-df["DAYS_EMPLOYED"] / 365.25).round(1)

# Create affordability-risk indicator
df["REPAYMENT_BURDEN"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
df["REPAYMENT_BURDEN"] = df["REPAYMENT_BURDEN"].replace([np.inf, -np.inf], np.nan)

# Create age groups for fairness auditing
df["AGE_GROUP"] = pd.cut(
    df["AGE_YEARS"],
    bins=[18, 30, 40, 50, 60, 100],
    labels=["18-30", "31-40", "41-50", "51-60", "60+"],
    include_lowest=True
)

# Create income groups for fairness auditing
df["INCOME_GROUP"] = pd.qcut(
    df["AMT_INCOME_TOTAL"],
    q=4,
    labels=["Low", "Medium-Low", "Medium-High", "High"],
    duplicates="drop"
)

# Save fairness reference file before encoding
fairness_columns = [
    "SK_ID_CURR",
    "TARGET",
    "CODE_GENDER",
    "AGE_GROUP",
    "INCOME_GROUP",
    "NAME_FAMILY_STATUS",
    "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE",
    "REPAYMENT_BURDEN"
]

fairness_columns = [col for col in fairness_columns if col in df.columns]
fairness_reference = df[fairness_columns].copy()
fairness_reference.to_csv("data/processed/fairness_reference.csv", index=False)

# Drop original day columns after creating meaningful versions
df = df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED"], errors="ignore")

# Separate target and ID
target = df["TARGET"]
ids = df["SK_ID_CURR"]

df_features = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")

# Identify categorical and numerical columns
categorical_columns = df_features.select_dtypes(include=["object", "category", "string"]).columns
numeric_columns = df_features.select_dtypes(include=["int64", "float64"]).columns

# Fill missing numerical values with median
for col in numeric_columns:
    df_features[col] = df_features[col].fillna(df_features[col].median())

# Fill missing categorical values with "Unknown"
for col in categorical_columns:
    df_features[col] = df_features[col].astype("object").fillna("Unknown")

# Encode categorical variables for machine learning
df_encoded = pd.get_dummies(df_features, columns=categorical_columns, drop_first=True, dtype=int)

# Add ID and target back
prepared_df = pd.concat([ids.reset_index(drop=True), target.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)

# Save prepared dataset
prepared_df.to_csv("data/processed/prepared_credit_data.csv", index=False)

# Save preparation summary
summary = pd.DataFrame({
    "Item": [
        "Original rows",
        "Rows after cleaning",
        "Original selected columns",
        "Prepared columns after encoding",
        "Prepared dataset file",
        "Fairness reference file"
    ],
    "Value": [
        original_rows,
        len(prepared_df),
        len(selected_columns),
        prepared_df.shape[1],
        "data/processed/prepared_credit_data.csv",
        "data/processed/fairness_reference.csv"
    ]
})

summary.to_csv("results/tables/data_preparation_summary.csv", index=False)

print("\nDATA PREPARATION COMPLETE")
print("-------------------------")
print(f"Prepared dataset shape: {prepared_df.shape}")
print("Prepared dataset saved to: data/processed/prepared_credit_data.csv")
print("Fairness reference saved to: data/processed/fairness_reference.csv")
print("Summary saved to: results/tables/data_preparation_summary.csv")