import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================= LOAD FILES =================
lr = joblib.load("model/logistic_regression.pkl")
dt = joblib.load("model/decision_tree.pkl")
knn = joblib.load("model/knn.pkl")
nb = joblib.load("model/naive_bayes.pkl")
rf = joblib.load("model/random_forest.pkl")
xgb = joblib.load("model/xgboost.pkl")

scaler = joblib.load("model/scaler.pkl")
model_columns = joblib.load("model/columns.pkl")

models = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "kNN": knn,
    "Naive Bayes": nb,
    "Random Forest": rf,
    "XGBoost": xgb
}

# ================= UI =================
st.title("💰 Adult Income Prediction")
st.write("Predict whether income is >50K or <=50K")

# -------- USER INPUT --------
age = st.slider("Age", 18, 90, 30)
hours = st.slider("Hours per week", 1, 99, 40)
education_num = st.slider("Education Number", 1, 16, 10)
capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.number_input("Capital Loss", 0, 5000, 0)

workclass = st.selectbox("Workclass", [
    "Private","Self-emp-not-inc","Self-emp-inc","Federal-gov",
    "Local-gov","State-gov","Without-pay","Never-worked"
])

marital_status = st.selectbox("Marital Status", [
    "Never-married","Married-civ-spouse","Divorced","Separated",
    "Widowed","Married-spouse-absent"
])

occupation = st.selectbox("Occupation", [
    "Tech-support","Craft-repair","Other-service","Sales","Exec-managerial",
    "Prof-specialty","Handlers-cleaners","Machine-op-inspct",
    "Adm-clerical","Farming-fishing","Transport-moving","Priv-house-serv",
    "Protective-serv","Armed-Forces"
])

gender = st.selectbox("Gender", ["Male","Female"])

race = st.selectbox("Race", [
    "White","Black","Asian-Pac-Islander","Amer-Indian-Eskimo","Other"
])

country = st.selectbox("Native Country", ["United-States","India","Other"])

model_choice = st.selectbox("Choose Model", list(models.keys()))

# ================= PREDICTION =================
if st.button("Predict Income"):

    input_dict = {
        "age": age,
        "hours-per-week": hours,
        "education-num": education_num,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        f"workclass_{workclass}": 1,
        f"marital-status_{marital_status}": 1,
        f"occupation_{occupation}": 1,
        f"gender_{gender}": 1,
        f"race_{race}": 1,
        f"native-country_{country}": 1
    }

    # Convert to dataframe
    input_df = pd.DataFrame([input_dict])

    # Align columns with training data
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    model = models[model_choice]
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # Output
    if prediction == 1:
        st.success(f"Predicted Income: >50K 💰 (Confidence: {prob:.2f})")
    else:
        st.info(f"Predicted Income: <=50K (Confidence: {1-prob:.2f})")
