import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Employee-Attrition - Employee-Attrition.csv")

st.set_page_config(page_title="Dashboard", layout='wide')

st.sidebar.title("Employee Attrition Analysis")
page = st.sidebar.radio("", ["Home", "Predict Employee Attrition"])

if page == "Home":
    st.title("Dashboard Home")
    st.header("Employee Insights Dashboard")

    col1, col2, col3 = st.columns(3)

    total_employees = len(df)
    attrition_yes = (df["Attrition"] == "Yes").sum()
    attrition_rate = (attrition_yes / total_employees) * 100
    performance_rate = df["PerformanceRating"].mean()

    with col1:
        st.subheader("High-Risk Employees")
        st.metric("Attrition", f"{attrition_rate:.2f}%")
        high_risk_employees = df[df["Attrition"] == "Yes"]
        st.dataframe(high_risk_employees[["EmployeeNumber", "Attrition"]])

    with col2:
        st.subheader("High Job Satisfaction")
        high_satisfaction = df[df['JobSatisfaction'] >= 4]
        st.dataframe(high_satisfaction[['EmployeeNumber', 'JobSatisfaction']])

    with col3:
        st.subheader("High Performance Score")
        if "PerformanceRating" in df.columns:
            high_perf = df.sort_values('PerformanceRating', ascending=False).head(5)
            st.dataframe(high_perf[['EmployeeNumber', 'PerformanceRating']])

    st.write("### Employee Data Table")
    st.dataframe(df)

elif page == "Predict Employee Attrition":
    st.title("Predict Employee Attrition")

    # Load trained objects
    model = joblib.load("model.pkl")
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_order = joblib.load("feature_order.pkl")
    gbc = joblib.load("gbc_model.pkl")   # Gradient Boosting model
    scaler = joblib.load("scaler_gb.pkl")  # Scaler used during training

    categorical_cols_encoder = encoder.feature_names_in_.tolist()

    frequent_values = {
        'BusinessTravel': 'Travel_Rarely',
        'OverTime': 'No',
        'Department': 'Sales',
        'EducationField': 'Life Sciences',
        'Gender': 'Male',
        'JobRole': 'Sales Executive',
        'TenureCategory': 'Experienced',
        'MaritalStatus': 'Married'
    }

    user_input = {}

    # --- Collect categorical inputs ---
    for col in categorical_cols_encoder:
        if col in df.columns:
            options = sorted(df[col].dropna().unique().tolist())
        else:
            options = [frequent_values.get(col, "missing")]
        default_value = frequent_values.get(col, options[0])
        user_input[col] = st.selectbox(
            col, options,
            index=options.index(default_value) if default_value in options else 0
        )

    # --- Collect numeric inputs ---
    numeric_cols = [col for col in feature_order if col not in categorical_cols_encoder]
    for col in numeric_cols:
        default_val = float(df[col].mean()) if col in df.columns else 0.0
        user_input[col] = st.number_input(col, value=default_val)

    # --- Build DataFrame ---
    df_input = pd.DataFrame([user_input])

    # Ensure all categorical columns exist
    for col in categorical_cols_encoder:
        if col not in df_input.columns:
            df_input[col] = frequent_values.get(col, "missing")

    # Encode categoricals
    df_categorical = df_input[categorical_cols_encoder]
    df_categorical_encoded = pd.DataFrame(
        encoder.transform(df_categorical),
        columns=encoder.get_feature_names_out(),
        index=df_categorical.index
    )

    # Separate numerics
    df_numerical = df_input[numeric_cols]

    # Final combined input
    df_final = pd.concat(
        [df_numerical.reset_index(drop=True), df_categorical_encoded.reset_index(drop=True)],
        axis=1
    )

    # Match training column order
    df_final = df_final.reindex(columns=feature_order, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df_final)

    # Debug: show inputs
    st.write("👉 User Input (raw):")
    st.dataframe(df_input)

    st.write("👉 Final feature vector before scaling:")
    st.dataframe(df_final)

    st.write("👉 Scaled Features:")
    st.dataframe(pd.DataFrame(df_scaled[0], index=df_final.columns, columns=['Scaled Value']))

    # --- Prediction ---
   # --- Prediction Section ---
st.write("### 🔮 Predict Employee Attrition")

xgb_model = joblib.load("xgb_final_model.pkl")

threshold = st.slider("Set Attrition Threshold", 0.0, 1.0, 0.5)

# ✅ Initialize variables safely before button click
probability = None
prediction = None

if st.button("Predict Attrition"):

    
    probability = xgb_model.predict_proba(df_scaled)[0][1]
    prediction = 1 if probability > threshold else 0

    st.metric(label="Attrition Probability", value=f"{probability*100:.2f}%")
    st.progress(min(int(probability * 100), 100))

    if prediction == 1:
        st.error(
            f"🚨 Employee has **left** the company.\n\n"
            f"🚨 Attrition Likely!\n"
            f"🧮 Probability: {probability:.2f}\n"
            f"🎯 Threshold: {threshold}"
        )
    else:
        st.success(
            f"✅ Employee is **still in** the company.\n\n"
            f"✅ No Attrition\n"
            f"🧮 Probability: {probability:.2f}\n"
            f"🎯 Threshold: {threshold}"
        )
