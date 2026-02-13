import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Adult Income Classification App")

st.write("Upload test dataset and select model for prediction.")

# Model selection
model_option = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "Naive Bayes", "XGBoost"]
)

# Load model
if model_option == "Logistic Regression":
    model = joblib.load("model/logistic.pkl")
elif model_option == "Decision Tree":
    model = joblib.load("model/decision_tree.pkl")
elif model_option == "Naive Bayes":
    model = joblib.load("model/naive_bayes.pkl")
elif model_option == "XGBoost":
    model = joblib.load("model/xgboost.pkl")

scaler = joblib.load("model/scaler.pkl")

# Upload CSV
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "income" in data.columns:
        X = data.drop("income", axis=1)
        y = data["income"]
    else:
        st.error("Dataset must contain 'income' column as target.")
        st.stop()

    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)

    # Evaluation Metrics
    acc = accuracy_score(y, predictions)
    prec = precision_score(y, predictions)
    rec = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    mcc = matthews_corrcoef(y, predictions)

    try:
        auc = roc_auc_score(y, model.predict_proba(X_scaled)[:,1])
    except:
        auc = "Not available"

    st.subheader("Evaluation Metrics")

    st.write(f"Accuracy: {acc:.3f}")
    st.write(f"Precision: {prec:.3f}")
    st.write(f"Recall: {rec:.3f}")
    st.write(f"F1 Score: {f1:.3f}")
    st.write(f"AUC: {auc}")
    st.write(f"MCC: {mcc:.3f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
