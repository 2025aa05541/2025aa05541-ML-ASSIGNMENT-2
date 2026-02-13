import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ---------------- PAGE TITLE ----------------
st.set_page_config(page_title="Adult Income Prediction", layout="wide")
st.title("💰 Adult Income Classification App")
st.write("Upload test CSV file to evaluate the selected model.")

# ---------------- LOAD PREPROCESSING FILES ----------------
scaler = joblib.load("model/scaler.pkl")
columns = joblib.load("model/columns.pkl")

# ---------------- SAFE MODEL LOADING ----------------
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        return None

# ---------------- LOAD MODELS ----------------
models = {
    "Logistic Regression": load_model("model/logistic_regression.pkl"),
    "Decision Tree": load_model("model/decision_tree.pkl"),
    "Naive Bayes": load_model("model/naive_bayes.pkl"),
    "Random Forest": load_model("model/random_forest.pkl"),
    "XGBoost": load_model("model/xgboost.pkl"),
    "kNN": load_model("model/knn.pkl")  # load your uploaded kNN model
}

# ---------------- MODEL SELECTION ----------------
model_name = st.selectbox("Select ML Model", list(models.keys()))

# ---------------- DATASET UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # -------- CLEAN COLUMN NAMES --------
    df.columns = df.columns.astype(str).str.strip()
    df.columns = df.columns.str.replace(".", "", regex=False)

    # -------- TARGET DETECTION --------
    target_col = df.columns[-1]
    X = df.drop(target_col, axis=1)
    y = df[target_col].astype(str)

    # Clean target values
    y = y.str.strip().str.replace(".", "", regex=False).str.replace(" ", "")
    y = y.map({"<=50K": 0, ">50K": 1})

    # Remove rows that failed mapping
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]

    # -------- ONE HOT ENCODING --------
    X = pd.get_dummies(X)

    # -------- ALIGN COLUMNS WITH TRAINING --------
    # Add missing columns from training
    for col in columns:
        if col not in X:
            X[col] = 0

    # Keep only training columns in correct order
    X = X[[col for col in columns if col in X.columns]]

    # -------- ENSURE NUMERIC --------
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    # -------- ROBUST PREPROCESSING FUNCTION --------
def preprocess_test_data(df, target_col, training_columns):
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col].astype(str)

    # Clean target
    y = y.str.strip().str.replace(".", "", regex=False).str.replace(" ", "")
    y = y.map({"<=50K": 0, ">50K": 1})
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]

    # One-hot encoding
    X = pd.get_dummies(X)

    # Add missing training columns
    for col in training_columns:
        if col not in X:
            X[col] = 0

    # Keep only training columns in order
    X = X[[col for col in training_columns if col in X.columns]]

    # Convert all to numeric and fill NaNs
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    return X, y


    # -------- SCALING --------
    X, y = preprocess_test_data(df, target_col, columns)
    X_scaled = scaler.transform(X)

    # -------- MODEL SELECTION --------
    model = models[model_name]

    if model is None:
        st.error(f"The selected model '{model_name}' is not available.")
    else:
        # -------- PREDICTIONS --------
        preds = model.predict(X_scaled)

        try:
            probs = model.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y, probs)
        except:
            auc = 0.0

        # -------- METRICS --------
        acc = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        mcc = matthews_corrcoef(y, preds)

        # ---------------- DISPLAY METRICS ----------------
        st.subheader("📊 Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.3f}")
        col1.metric("Precision", f"{prec:.3f}")
        col2.metric("Recall", f"{rec:.3f}")
        col2.metric("F1 Score", f"{f1:.3f}")
        col3.metric("AUC", f"{auc:.3f}")
        col3.metric("MCC", f"{mcc:.3f}")

        # ---------------- CONFUSION MATRIX ----------------
        st.subheader("🔎 Confusion Matrix")
        cm = confusion_matrix(y, preds)
        st.write(cm)

        # ---------------- CLASSIFICATION REPORT ----------------
        st.subheader("📄 Classification Report")
        st.text(classification_report(y, preds))

else:
    st.info("Please upload a CSV file to proceed.")

