
import streamlit as st
import pickle
import numpy as np

# Load scaler and model
with open("/content/drive/MyDrive/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("/content/drive/MyDrive/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Loan Default Risk Predictor", layout="centered")

st.title("💸 Loan Default Risk Prediction")
st.markdown("Enter loan application details to predict risk of default.")

# Define input fields
amount = st.number_input("Loan Amount", min_value=0.0)
payment = st.number_input("Monthly Payment", min_value=0.0)
income = st.number_input("Annual Income", min_value=0.0)
debtIncRat = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0)
installment_to_income = payment / income if income > 0 else 0
loan_to_income = amount / income if income > 0 else 0

# Add more fields as needed
totalPaid = st.number_input("Total Paid", min_value=0.0)
revolRatio = st.number_input("Revolving Ratio", min_value=0.0, max_value=1.0)
bcRatio = st.number_input("Bankcard Utilization Ratio", min_value=0.0, max_value=1.0)
totalAcc = st.number_input("Total Accounts", min_value=0)
totalRevLim = st.number_input("Total Revolving Limit", min_value=0.0)

if st.button("Predict"):
    # Assemble features into the correct order
    features = np.array([[ 
        amount, payment, debtIncRat, totalPaid, revolRatio, totalAcc, 
        totalRevLim, bcRatio, loan_to_income, installment_to_income
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    if pred == 1:
        st.error(f" High Risk of Default! (Probability: {prob:.2f})")
    else:
        st.success(f"Low Risk of Default (Probability: {prob:.2f})")
