import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Responsible AI Credit Scoring Dashboard",
    page_icon="💳",
    layout="wide"
)


# -----------------------------
# Project paths
# -----------------------------
BASE_DIR = BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "class_weighted_logistic_regression_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "final_model_feature_columns.pkl"

RESULTS_DIR = BASE_DIR / "results"

FINAL_DIR = RESULTS_DIR / "final_comparison"
FINAL_TABLES_DIR = FINAL_DIR / "tables"
FINAL_CHARTS_DIR = FINAL_DIR / "charts"

AFFORDABILITY_DIR = RESULTS_DIR / "affordability"
AFFORDABILITY_TABLES_DIR = AFFORDABILITY_DIR / "tables"
AFFORDABILITY_CHARTS_DIR = AFFORDABILITY_DIR / "charts"

EXPLAINABILITY_DIR = RESULTS_DIR / "explainability"
EXPLAINABILITY_TABLES_DIR = EXPLAINABILITY_DIR / "tables"
EXPLAINABILITY_CHARTS_DIR = EXPLAINABILITY_DIR / "charts"

SHAP_DIR = RESULTS_DIR / "shap"
SHAP_TABLES_DIR = SHAP_DIR / "tables"
SHAP_CHARTS_DIR = SHAP_DIR / "charts"

APP_TESTING_DIR = RESULTS_DIR / "app_testing"


# -----------------------------
# Load model and features
# -----------------------------
@st.cache_resource
def load_model_and_features():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, feature_columns


model, feature_columns = load_model_and_features()


# -----------------------------
# Helper functions
# -----------------------------
def first_existing_path(paths):
    """
    Accepts a single path or list of paths.
    Returns the first path that exists.
    If none exists, returns the first path so Streamlit can show a useful warning.
    """
    if isinstance(paths, (str, Path)):
        return Path(paths)

    for path in paths:
        path = Path(path)
        if path.exists():
            return path

    return Path(paths[0])


def build_input_row(user_inputs, feature_columns):
    row = pd.DataFrame(
        np.full((1, len(feature_columns)), np.nan),
        columns=feature_columns
    )

    for feature, value in user_inputs.items():
        if feature in row.columns:
            row.loc[0, feature] = value

    return row


def add_one_hot_feature(user_inputs, feature_name):
    if feature_name in feature_columns:
        user_inputs[feature_name] = 1


def decision_label(risk_probability, repayment_burden):
    if risk_probability >= 0.60:
        return "High repayment difficulty risk", "Further review recommended"
    elif repayment_burden >= 0.35:
        return "Moderate risk with high affordability pressure", "Affordability review recommended"
    elif risk_probability >= 0.40:
        return "Moderate repayment difficulty risk", "Manual review recommended"
    else:
        return "Lower repayment difficulty risk", "Likely acceptable in demo context"


def display_csv(title, file_path):
    st.subheader(title)

    file_path = first_existing_path(file_path)

    if file_path.exists():
        df = pd.read_csv(file_path)
        st.dataframe(df, width="stretch")
    else:
        st.warning(f"File not found: {file_path}")


def display_image(title, file_path):
    st.subheader(title)

    file_path = first_existing_path(file_path)

    if file_path.exists():
        st.image(str(file_path), width="stretch")
    else:
        st.warning(f"Image not found: {file_path}")


def display_text_file(title, file_path):
    st.subheader(title)

    file_path = first_existing_path(file_path)

    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        st.text(text)
    else:
        st.warning(f"Text file not found: {file_path}")


# -----------------------------
# Main title
# -----------------------------
st.title("Responsible AI Credit Scoring Dashboard")

st.write(
    "This dashboard presents the project artefact for a fairness-aware and affordability-aware "
    "machine learning framework for credit scoring. It includes a single-applicant demo, model "
    "performance results, fairness analysis, affordability-risk analysis, explainability outputs, "
    "and app testing evidence."
)

st.info(
    ""
)


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Single Applicant Demo",
        "Model Performance",
        "Fairness Audit",
        "Affordability Analysis",
        "Explainability / SHAP",
        "App Testing Evidence",
        "Final Project Summary"
    ]
)


# =====================================================
# TAB 1: SINGLE APPLICANT DEMO
# =====================================================
with tab1:
    st.header("Single Applicant Credit Scoring Demo")

    st.write(
        "This section allows a user to enter applicant details and receive a repayment difficulty "
        "prediction, repayment burden calculation, affordability warning, and simple explanation."
    )

    st.info(
        "Model used: Class-weighted Logistic Regression. "
        "TARGET = 1 means repayment difficulty. "
        "The high repayment-burden threshold used in this project is 0.35."
    )

    input_errors = []
    input_warnings = []

    st.subheader("Applicant Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        income = st.number_input(
            "Annual income",
            min_value=0.0,
            max_value=100000000.0,
            value=150000.0,
            step=1000.0,
            help="Enter the applicant's annual income."
        )

        if income <= 0:
            st.error("Annual income must be greater than 0.")
            input_errors.append("Annual income must be greater than 0.")

        credit_amount = st.number_input(
            "Credit amount requested",
            min_value=0.0,
            max_value=100000000.0,
            value=500000.0,
            step=1000.0,
            help="Enter the requested credit amount."
        )

        if credit_amount <= 0:
            st.error("Credit amount requested must be greater than 0.")
            input_errors.append("Credit amount requested must be greater than 0.")

        annuity = st.number_input(
            "Annual repayment / annuity amount",
            min_value=0.0,
            max_value=100000000.0,
            value=30000.0,
            step=500.0,
            help="Enter the annual repayment or annuity amount."
        )

        if annuity <= 0:
            st.error("Annual repayment / annuity amount must be greater than 0.")
            input_errors.append("Annual repayment / annuity amount must be greater than 0.")

        goods_price = st.number_input(
            "Goods price",
            min_value=0.0,
            max_value=100000000.0,
            value=450000.0,
            step=1000.0,
            help="Enter the price of goods linked to the credit application."
        )

        if goods_price <= 0:
            st.error("Goods price must be greater than 0.")
            input_errors.append("Goods price must be greater than 0.")

    with col2:
        age_years = st.number_input(
            "Age",
            min_value=0.0,
            max_value=150.0,
            value=35.0,
            step=1.0,
            help="Enter applicant age."
        )

        if age_years < 18 or age_years > 80:
            st.error("Age must be between 18 and 80 for this demo.")
            input_errors.append("Age must be between 18 and 80 for this demo.")

        employment_years = st.number_input(
            "Employment years",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=1.0,
            help="Enter number of years employed."
        )

        if employment_years > max(age_years - 16, 0):
            st.error("Employment years cannot be greater than the applicant's realistic working-age period.")
            input_errors.append("Employment years cannot realistically be greater than the applicant's working-age period.")

        ext_source_2 = st.slider(
            "External source score 2",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.01,
            help="A higher external source score generally indicates lower repayment difficulty risk."
        )

        ext_source_3 = st.slider(
            "External source score 3",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.01,
            help="A higher external source score generally indicates lower repayment difficulty risk."
        )

        family_members = st.number_input(
            "Number of family members",
            min_value=0.0,
            max_value=100000000.0,
            value=2.0,
            step=1.0,
            help="Enter number of family members."
        )

        if family_members < 1:
            st.error("Number of family members must be at least 1.")
            input_errors.append("Number of family members must be at least 1.")
        elif family_members > 20:
            st.error("Number of family members is unrealistically high. Use a realistic value between 1 and 20.")
            input_errors.append("Number of family members is unrealistically high.")

        children = st.number_input(
            "Number of children",
            min_value=0.0,
            max_value=100000000.0,
            value=0.0,
            step=1.0,
            help="Enter number of children."
        )

        if children > family_members:
            st.error("Number of children cannot be greater than number of family members.")
            input_errors.append("Number of children cannot be greater than number of family members.")
        elif children > 20:
            st.error("Number of children is unrealistically high. Use a realistic value between 0 and 20.")
            input_errors.append("Number of children is unrealistically high.")

    with col3:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        owns_car = st.selectbox(
            "Owns car?",
            ["No", "Yes"]
        )

        owns_realty = st.selectbox(
            "Owns realty/property?",
            ["No", "Yes"]
        )

        family_status = st.selectbox(
            "Family status",
            [
                "Unknown",
                "Married",
                "Single / not married",
                "Civil marriage",
                "Separated",
                "Widow"
            ]
        )

        income_type = st.selectbox(
            "Income type",
            [
                "Unknown",
                "Working",
                "Commercial associate",
                "Pensioner",
                "State servant",
                "Student",
                "Unemployed"
            ]
        )

        education_type = st.selectbox(
            "Education type",
            [
                "Unknown",
                "Secondary / secondary special",
                "Higher education",
                "Incomplete higher",
                "Lower secondary",
                "Academic degree"
            ]
        )

        occupation = st.selectbox(
            "Occupation type",
            [
                "Unknown",
                "Laborers",
                "Drivers",
                "Sales staff",
                "Core staff",
                "Managers",
                "High skill tech staff",
                "Accountants",
                "Medicine staff",
                "Security staff",
                "Cooking staff",
                "Cleaning staff"
            ]
        )

    # -----------------------------
    # Derived affordability feature
    # -----------------------------
    repayment_burden = annuity / income if income > 0 else np.nan

    st.subheader("Derived affordability indicator")

    if pd.isna(repayment_burden):
        st.write("Repayment burden = **Not available because income is 0**")
    else:
        st.write(f"Repayment burden = **{repayment_burden:.3f}**")

    if pd.notna(repayment_burden) and repayment_burden >= 0.35:
        st.warning("High repayment burden detected. This applicant may require affordability review.")
    elif pd.notna(repayment_burden):
        st.success("Repayment burden is below the project high-burden threshold of 0.35.")

    # -----------------------------
    # Cross-field warnings
    # -----------------------------
    if income > 0 and annuity > income:
        st.warning(
            "Extreme affordability-risk case: annual repayment / annuity amount is higher than annual income. "
            "This does not automatically block the prediction, but it should be reviewed carefully."
        )
        input_warnings.append(
            "Annual repayment / annuity amount is higher than annual income, indicating extreme affordability pressure."
        )

    if pd.notna(repayment_burden) and repayment_burden > 2:
        st.warning(
            "Extreme repayment-burden case: repayments are more than twice the applicant's income. "
            "This does not automatically block the prediction, but it should be reviewed carefully."
        )
        input_warnings.append(
            "Repayment burden is above 2.0, meaning repayments are more than twice the applicant's income."
        )

    if income > 0 and credit_amount > income * 20:
        st.warning(
            "Extreme credit-risk case: credit amount is very high compared with annual income. "
            "This does not automatically block the prediction, but it should be reviewed carefully."
        )
        input_warnings.append(
            "Credit amount is very high compared with annual income."
        )

    if credit_amount > 0 and goods_price > credit_amount * 2:
        st.warning(
            "Unusual goods-price case: goods price is unusually high compared with the credit amount. "
            "This does not automatically block the prediction, but it should be reviewed carefully."
        )
        input_warnings.append(
            "Goods price is unusually high compared with the credit amount."
        )

    if credit_amount < 1000:
        input_warnings.append("Credit amount is very low compared with typical credit application values.")

    if goods_price < 1000:
        input_warnings.append("Goods price is very low compared with typical training data values.")

    if annuity < 100:
        input_warnings.append("Annual repayment / annuity amount is very low.")

    # -----------------------------
    # Build model input
    # -----------------------------
    user_inputs = {
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit_amount,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price,
        "AGE_YEARS": age_years,
        "EMPLOYMENT_YEARS": employment_years,
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
        "CNT_FAM_MEMBERS": family_members,
        "CNT_CHILDREN": children,
        "REPAYMENT_BURDEN": repayment_burden,
        "CODE_GENDER_M": 1 if gender == "Male" else 0,
        "FLAG_OWN_CAR_Y": 1 if owns_car == "Yes" else 0,
        "FLAG_OWN_REALTY_Y": 1 if owns_realty == "Yes" else 0,
    }

    if family_status != "Unknown":
        add_one_hot_feature(user_inputs, f"NAME_FAMILY_STATUS_{family_status}")

    if income_type != "Unknown":
        add_one_hot_feature(user_inputs, f"NAME_INCOME_TYPE_{income_type}")

    if education_type != "Unknown":
        add_one_hot_feature(user_inputs, f"NAME_EDUCATION_TYPE_{education_type}")

    if occupation != "Unknown":
        add_one_hot_feature(user_inputs, f"OCCUPATION_TYPE_{occupation}")

    input_row = build_input_row(user_inputs, feature_columns)

    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button("Run Credit Scoring Demo"):
        if input_errors:
            st.error("Prediction blocked. Please correct the unrealistic input value(s) shown above.")
            st.stop()

        if input_warnings:
            st.warning("Some values are unusual. The app can still run, but interpret the result carefully:")
            for warning in input_warnings:
                st.write(f"- {warning}")

        risk_probability = model.predict_proba(input_row)[0, 1]
        prediction = model.predict(input_row)[0]

        risk_label, recommendation = decision_label(risk_probability, repayment_burden)

        st.header("Prediction Result")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric("Predicted repayment difficulty probability", f"{risk_probability:.2%}")

        with metric_col2:
            st.metric(
                "Model prediction",
                "Repayment difficulty" if prediction == 1 else "No repayment difficulty"
            )

        with metric_col3:
            st.metric("Repayment burden", f"{repayment_burden:.3f}")

        st.subheader("Demo decision interpretation")
        st.write(f"**Risk level:** {risk_label}")
        st.write(f"**Recommendation:** {recommendation}")

        st.subheader("Simple explanation")

        explanation_points = []

        if repayment_burden >= 0.35:
            explanation_points.append(
                "The applicant has high repayment burden, which indicates possible affordability pressure."
            )
        else:
            explanation_points.append(
                "The applicant's repayment burden is below the project high-burden threshold."
            )

        if ext_source_2 < 0.40 or ext_source_3 < 0.40:
            explanation_points.append(
                "One or more external source scores are low, which can increase predicted repayment difficulty risk."
            )
        elif ext_source_2 > 0.60 and ext_source_3 > 0.60:
            explanation_points.append(
                "The external source scores are relatively strong, which can reduce predicted repayment difficulty risk."
            )
        else:
            explanation_points.append(
                "The external source scores are in a moderate range."
            )

        if credit_amount > income * 4:
            explanation_points.append(
                "The requested credit amount is high compared with income."
            )

        if employment_years < 1:
            explanation_points.append(
                "Short employment duration may increase uncertainty in the applicant profile."
            )

        for point in explanation_points:
            st.write(f"- {point}")

        st.write(
            "This prediction is influenced by factors such as income, credit amount, annuity amount, "
            "repayment burden, external source scores, employment years, age, and selected borrower profile variables. "
            "The model used here is class-weighted Logistic Regression, selected because it improves repayment difficulty "
            "detection compared with the baseline models."
        )

        if repayment_burden >= 0.35:
            st.info(
                "The applicant has high repayment burden under the project threshold. "
                "This does not automatically mean rejection, but it indicates that affordability should be reviewed."
            )

        with st.expander("View model-ready input row"):
            st.dataframe(input_row, width="stretch")


# =====================================================
# TAB 2: MODEL PERFORMANCE
# =====================================================
with tab2:
    st.header("Model Performance Comparison")

    st.write(
        "This section compares the baseline and improved models using predictive performance metrics. "
        "The project showed that accuracy alone was not enough because the dataset was highly imbalanced."
    )

    display_csv(
        "Final Model Performance Comparison Table",
        FINAL_TABLES_DIR / "final_model_performance_comparison.csv"
    )

    col_a, col_b = st.columns(2)

    with col_a:
        display_image(
            "Recall Comparison",
            FINAL_CHARTS_DIR / "final_recall_comparison.png"
        )

    with col_b:
        display_image(
            "Accuracy Comparison",
            FINAL_CHARTS_DIR / "final_accuracy_comparison.png"
        )

    st.success(
        "Key interpretation: the class-weighted Logistic Regression model improved repayment difficulty recall, "
        "but this came with reduced accuracy and precision. This supports the project argument that responsible "
        "credit scoring requires trade-off analysis, not accuracy alone."
    )


# =====================================================
# TAB 3: FAIRNESS AUDIT
# =====================================================
with tab3:
    st.header("Fairness Audit and Trade-off Analysis")

    st.write(
        "This section presents group-level fairness audit results. The audit is diagnostic and does not prove "
        "discrimination. It identifies differences in model outcomes across borrower groups."
    )

    display_csv(
        "Final Fairness Trade-off Summary",
        FINAL_TABLES_DIR / "final_fairness_tradeoff_summary.csv"
    )

    display_image(
        "Fairness Approval Difference Comparison",
        FINAL_CHARTS_DIR / "final_fairness_approval_difference_comparison.png"
    )

    st.warning(
        "Key interpretation: improving repayment difficulty detection increased some group-level differences. "
        "This shows a responsible AI trade-off rather than a perfect fairness solution."
    )


# =====================================================
# TAB 4: AFFORDABILITY ANALYSIS
# =====================================================
with tab4:
    st.header("Affordability-Risk Analysis")

    st.write(
        "This section analyses affordability risk using repayment burden as a project-level proxy. "
        "Repayment burden was calculated as annual repayment divided by annual income."
    )

    display_csv(
        "Final Affordability-Risk Summary",
        FINAL_TABLES_DIR / "final_affordability_risk_summary.csv"
    )

    display_image(
        "High-Burden Difficulty Comparison",
        FINAL_CHARTS_DIR / "final_high_burden_difficulty_comparison.png"
    )

    display_csv(
        "Detailed Affordability Model Summary",
        AFFORDABILITY_TABLES_DIR / "model_affordability_risk_summary.csv"
    )

    col_c, col_d = st.columns(2)

    with col_c:
        display_image(
            "High-Burden Approvals by Model",
            AFFORDABILITY_CHARTS_DIR / "high_burden_approvals_by_model.png"
        )

    with col_d:
        display_image(
            "Average Repayment Burden Among Approvals",
            AFFORDABILITY_CHARTS_DIR / "average_repayment_burden_among_approvals.png"
        )

    st.success(
        "Key interpretation: the baseline models approved almost all high repayment-burden applicants, "
        "while the class-weighted model was more cautious. This supports the affordability-aware part "
        "of the framework."
    )


# =====================================================
# TAB 5: EXPLAINABILITY / SHAP
# =====================================================
with tab5:
    st.header("Explainability and SHAP Analysis")

    st.write(
        "This section presents feature importance, logistic regression coefficients, and SHAP-based explanations. "
        "The purpose is to show which variables influenced the model outputs."
    )

    display_csv(
        "Final Explainability Summary",
        FINAL_TABLES_DIR / "final_explainability_summary.csv"
    )

    display_csv(
        "Combined Top Explainability Features",
        EXPLAINABILITY_TABLES_DIR / "combined_top_explainability_features.csv"
    )

    st.subheader("Model Explainability Charts")

    col_e, col_f = st.columns(2)

    with col_e:
        display_image(
            "Random Forest Top Feature Importance",
            EXPLAINABILITY_CHARTS_DIR / "random_forest_top_feature_importance.png"
        )

    with col_f:
        display_image(
            "Class-weighted Logistic Regression Top Coefficients",
            EXPLAINABILITY_CHARTS_DIR / "class_weighted_logistic_regression_top_coefficients.png"
        )

    st.subheader("SHAP Outputs")

    display_csv(
        "Combined Top SHAP Features",
        SHAP_TABLES_DIR / "combined_top_shap_features.csv"
    )

    col_g, col_h = st.columns(2)

    with col_g:
        display_image(
            "Random Forest SHAP Summary Plot",
            SHAP_CHARTS_DIR / "random_forest_baseline_shap_summary_plot.png"
        )

    with col_h:
        display_image(
            "Class-weighted Logistic Regression SHAP Summary Plot",
            SHAP_CHARTS_DIR / "class_weighted_logistic_regression_shap_summary_plot.png"
        )

    st.success(
        "Key interpretation: repayment burden, external source scores, credit amount, goods price, age, "
        "and employment-related features were among the important variables. This supports the explainability "
        "component of the responsible AI framework."
    )


# =====================================================
# TAB 6: APP TESTING EVIDENCE
# =====================================================
with tab6:
    st.header("App Testing Evidence")

    st.write(
        "This section shows real test cases exported from the test split. These cases were used to test "
        "the Streamlit app with examples linked to actual TARGET values."
    )

    display_csv(
        "Real Dataset App Test Cases",
        APP_TESTING_DIR / "real_dataset_app_test_cases.csv"
    )

    display_text_file(
        "App Testing Notes",
        APP_TESTING_DIR / "app_testing_notes.txt"
    )

    screenshots_dir = APP_TESTING_DIR / "screenshots"

    st.subheader("Screenshot Evidence Folder")

    if screenshots_dir.exists():
        screenshot_files = sorted(
            list(screenshots_dir.glob("*.png")) +
            list(screenshots_dir.glob("*.jpg")) +
            list(screenshots_dir.glob("*.jpeg"))
        )

        if screenshot_files:
            for screenshot in screenshot_files:
                st.image(str(screenshot), caption=screenshot.name, width="stretch")
        else:
            st.warning("Screenshot folder exists, but no image files were found.")
    else:
        st.warning(f"Screenshot folder not found: {screenshots_dir}")

    st.info(
        "The app testing evidence demonstrates low-risk, high-risk, high-affordability-risk, and invalid-input behaviour."
    )


# =====================================================
# TAB 7: FINAL PROJECT SUMMARY
# =====================================================
with tab7:
    st.header("Final Project Summary")

    st.write(
        "The project designed and evaluated a fairness-aware and affordability-aware machine learning "
        "framework for credit scoring. The aim was not only to predict repayment difficulty, but to "
        "evaluate whether responsible AI methods can provide better decision support than accuracy-focused "
        "credit scoring alone."
    )

    st.subheader("What was built")

    st.write(
        "- A prepared borrower-level credit scoring dataset.\n"
        "- Baseline Logistic Regression and Random Forest models.\n"
        "- A class-weighted Logistic Regression model to improve repayment difficulty detection.\n"
        "- A fairness audit across borrower groups.\n"
        "- An affordability-risk analysis using repayment burden.\n"
        "- Explainability analysis using coefficients, feature importance, and SHAP.\n"
        "- A Streamlit dashboard demonstrating the final responsible AI framework."
    )

    st.subheader("Main finding")

    st.write(
        "The results showed that traditional accuracy-focused evaluation can be misleading in an imbalanced "
        "credit scoring dataset. The baseline Logistic Regression model achieved high accuracy but detected "
        "very few repayment difficulty cases. The class-weighted Logistic Regression model improved repayment "
        "difficulty recall and was more cautious with affordability-risk cases, but this introduced trade-offs "
        "in accuracy, precision, and group-level fairness differences."
    )

    st.subheader("Research question answer")

    st.write(
        "The fairness-aware and affordability-aware framework improved the evaluation of credit scoring decisions "
        "by combining predictive performance, fairness auditing, affordability-risk analysis, and explainability. "
        "It did not create a perfect model, but it provided a more responsible and transparent way to assess model "
        "outputs than relying on accuracy alone."
    )

    st.subheader("Limitations")

    st.write(
        "- The dataset is public and historical, not live lender data.\n"
        "- Repayment burden is used as a proxy for affordability, not a full affordability assessment.\n"
        "- The fairness audit is diagnostic and does not prove discrimination.\n"
        "- The Streamlit app is an academic prototype, not a production lending system.\n"
        "- Real-world deployment would require legal compliance, stronger affordability checks, monitoring, security, and human review."
    )

    display_csv(
        "Final Responsible AI Comparison Table",
        FINAL_TABLES_DIR / "final_responsible_ai_comparison_table.csv"
    )

    display_text_file(
        "Final Research Question Answer Notes",
        [
            FINAL_TABLES_DIR / "final_research_question_answer_notes.txt",
            FINAL_DIR / "final_research_question_answer_notes.txt"
        ]
    )


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("About this dashboard")

st.sidebar.write("**Project:** Responsible AI Credit Scoring")

st.sidebar.write("**Model used in demo:** Class-weighted Logistic Regression")

st.sidebar.write("**TARGET = 1:** Repayment difficulty")

st.sidebar.write("**High repayment burden threshold:** 0.35")

st.sidebar.write(
    "This dashboard is an academic artefact for demonstrating model prediction, fairness auditing, "
    "affordability-risk analysis, explainability, and app testing evidence."
)