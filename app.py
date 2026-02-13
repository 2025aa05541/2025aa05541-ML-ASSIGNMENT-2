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
model_files = {
    "Logistic Regression": "model/logistic.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "XGBoost": "model/xgboost.pkl"
}

model = joblib.load(model_files[model_option])
scaler = joblib.load("model/scaler.pkl")

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Separate target
    y = df["income"]
    X = df.drop("income", axis=1)

    # SAME preprocessing as training
    X = pd.get_dummies(X)

    # Align columns with training scaler
    model_features = scaler.feature_names_in_
    X = X.reindex(columns=model_features, fill_value=0)

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    preds = model.predict(X_scaled)

    # Metrics
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    mcc = matthews_corrcoef(y, preds)

    try:
        auc = roc_auc_score(y, model.predict_proba(X_scaled)[:,1])
    except:
        auc = "Not Available"

    st.subheader("Evaluation Metrics")
    st.write(f"Accuracy: {acc:.3f}")
    st.write(f"Precision: {prec:.3f}")
    st.write(f"Recall: {rec:.3f}")
    st.write(f"F1 Score: {f1:.3f}")
    st.write(f"AUC: {auc}")
    st.write(f"MCC: {mcc:.3f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, preds)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
