import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model and column names
model, columns = joblib.load("model_xgb.pkl")

st.title("📉 Customer Churn Predictor")

st.markdown("""
This app predicts **whether a telecom customer will churn** based on their contract details, payment method, internet usage, etc.
""")

# Sample inputs for Streamlit
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", ["Yes", "No"])
Partner = st.selectbox("Has a Partner", ["Yes", "No"])
Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
MonthlyCharges = st.number_input("Monthly Charges", 10.0, 200.0, 70.0)
TotalCharges = st.number_input("Total Charges", 10.0, 8000.0, 1500.0)

# Other categorical inputs
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# Convert inputs to dataframe
input_dict = {
    "gender": gender,
    "SeniorCitizen": 1 if SeniorCitizen == "Yes" else 0,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
    "InternetService": InternetService,
    "Contract": Contract,
    "PaymentMethod": PaymentMethod
}

input_df = pd.DataFrame([input_dict])

# One-hot encode to match training features
input_df = pd.get_dummies(input_df)
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[columns]

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN.")
    else:
        st.success("✅ Customer is likely to STAY.")
