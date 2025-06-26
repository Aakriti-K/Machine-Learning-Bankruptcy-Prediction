# -*- coding: utf-8 -*-
'''
To open it in localhost, open the terminal and run the following prompts:
1. pip install streamlit
2. pip install joblib
3. python -m streamlit run app_py.py

You will get the URL. Click on it and enter the different values. Click the "Predict Bankruptcy Risk" button to get the overview of financial health and risk score

'''





import streamlit as st
import numpy as np
import joblib

# Load the saved model
model = joblib.load('best_bankruptcy_model.pkl')

# Define the top 15 features (as seen in the feature importance plot)
feature_names = [
    "Net Income to Stockholder's Equity",
 ' Net Value Growth Rate',
 ' Persistent EPS in the Last Four Seasons',
 ' Borrowing dependency',
 ' Non-industry income and expenditure/revenue',
 ' Per Share Net profit before tax (Yuan ¥)',
 ' Working Capital/Equity',
 ' Net profit before tax/Paid-in capital',
 ' Equity to Liability',
 ' Cash/Total Assets',
 ' Interest Expense Ratio',
 ' Net Value Per Share (B)',
 ' Net Income to Total Assets',
 ' Interest Coverage Ratio (Interest expense to EBIT)',
 ' Degree of Financial Leverage (DFL)'
]

# App title
st.title("💼 Bankruptcy Prediction App")
st.markdown("Enter the following financial indicators for a company:")

# Collect input for the 15 features
user_inputs = []
for feature in feature_names:
    val = st.number_input(f"{feature}:", format="%.6f")
    user_inputs.append(val)

# Predict on button click
if st.button("🔍 Predict Bankruptcy Risk"):
    input_array = np.array(user_inputs).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    prob = model.predict_proba(input_array)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk: The company is likely to go bankrupt.\n\n**Risk Score: {prob:.2f}**")
    else:
        st.success(f"✅ Low Risk: The company is financially healthy.\n\n**Risk Score: {prob:.2f}**")