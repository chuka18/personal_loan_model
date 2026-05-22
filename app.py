import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("loan_pipeline.pkl")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Personal Loan Predictor",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Personal Loan Prediction App")
st.write("Fill in customer details below to predict loan eligibility.")

# -----------------------------
# INPUT UI
# -----------------------------

age = st.slider("Age", 18, 80, 35)

experience = st.slider("Experience (Years)", 0, 15)

income = st.slider("Income (K$)", 0, 200, 50)

family = st.selectbox("Family Size", [1, 2, 3, 4])

ccavg = st.slider("CCAvg (Monthly Credit Card Spend in K$)", 0.0, 10.0, 1.5)

education = st.selectbox(
    "Education Level",
    ["Undergraduate", "Graduate", "Advanced/Professional"]
)

mortgage = st.slider("Mortgage (K$)", 0, 500, 0)

securities_account = st.toggle("Has Securities Account")

cd_account = st.toggle("Has CD Account")

online = st.toggle("Uses Online Banking")

credit_card = st.toggle("Has Credit Card")

# -----------------------------
# CONVERT BOOL TO 0/1
# -----------------------------
securities_account = int(securities_account)
cd_account = int(cd_account)
online = int(online)
credit_card = int(credit_card)

# -----------------------------
# CREATE DATAFRAME (IMPORTANT: EXACT COLUMN NAMES)
# -----------------------------
input_data = pd.DataFrame([[
    age,
    experience,
    income,
    family,
    ccavg,
    education,
    mortgage,
    securities_account,
    cd_account,
    online,
    credit_card
]], columns=[
    'Age','Experience','Income','Family','CCAvg','Education',
    'Mortgage','Securities Account','CD Account','Online','CreditCard'
])

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Loan Outcome 🚀"):
    input_data = pd.DataFrame([[ 
        age, experience, income, family, ccavg,
        education, mortgage,
        securities_account, cd_account, online, credit_card
    ]], columns=[
        'Age','Experience','Income','Family','CCAvg','Education',
        'Mortgage','Securities Account','CD Account','Online','CreditCard'
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✔ Customer WILL take loan (Probability: {probability:.2f})")
    else:
        st.error(f"✖ Customer will NOT take loan (Probability: {probability:.2f})")

# -----------------------------
# SHOW INPUT (DEBUG OPTION)
# -----------------------------
with st.expander("Show Input Data"):
    st.write(input_data)