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

from sklearn.neighbors import KNeighborsClassifier

# ---------------- PAGE TITLE ----------------
st.set_page_config(page_title="Adult Income Prediction", layout="wide")
st.title("💰 Adult Income Classification App")
st.write("Upload raw test CSV (unscaled) to evaluate your model.")

# ---------------- LOAD SCALER AND TRAINING COLUMNS ----------------
scaler = joblib.load("model/scaler.pkl")        # Your saved StandardScaler
columns = joblib.load("model/columns.pkl")      # Columns used in training

# ---------------- SAFE MODEL LOADING ----------------
def load_model(path):
    return joblib.load(path) if os.path.exists(path) else None

# ---------------- LOAD MODELS ----------------
models = {
    "Logistic Regression": load_model("model/logistic_regression.pkl"),
    "Decision Tree": load_model("model/decision_tree.pkl"),
    "Naive Bayes": load_model("model/naive_bayes.pkl"),
    "Random Forest": load_model("model/random_forest.pkl"),
    "XGBoost": load_model("model/xgboost.pkl"),
    "kNN": load_model("model/knn.pkl")  # your uploaded kNN model
}

# ---------------- MODEL SELECTION ----------------
model_name = st.selectbox("Select ML Model", list(models.keys()))

# ---------------- ROBUST PREPROCESSING ----------------
def preprocess_test_data(df, target_col, training_columns):
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col].astype(str)

    # Clean target
    y = y.str.strip().str.replace(".", "", regex=False).str.replace(" ", "")
    y = y.map({"<=50K": 0, ">50K": 1})

    # Keep only valid rows
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]

    # One-hot encode categorical features
    X = pd.get_dummies(X)

    # Add missing training columns
    for col in training_columns:
        if col not in X:
            X[col] = 0

    # Drop extra columns
    extra_cols = set(X.columns) - set(training_columns)
    if extra_cols:
        X = X.drop(columns=list(extra_cols))

    # Reorder columns exactly like training
    X = X[training_columns]

    # Convert all to numeric and fill NaNs
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    return X, y

# ---------------- DATA UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    target_col = df.columns[-1]  # assume last column is target

    # Preprocess and align columns
    X, y = preprocess_test_data(df, target_col, columns)

    # Scale features safely
    X_scaled = scaler.transform(X)

    # Load selected model
    model = models[model_name]
    if model is None:
        st.error(f"Model '{model_name}' not available.")
    else:
        # Make predictions
        preds = model.predict(X_scaled)
        try:
            probs = model.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y, probs)
        except:
            auc = 0.0

        # ---------------- METRICS ----------------
        acc = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        mcc = matthews_corrcoef(y, preds)

        # ---------------- DISPLAY ----------------
        st.subheader("📊 Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.3f}")
        col1.metric("Precision", f"{prec:.3f}")
        col2.metric("Recall", f"{rec:.3f}")
        col2.metric("F1 Score", f"{f1:.3f}")
        col3.metric("AUC", f"{auc:.3f}")
        col3.metric("MCC", f"{mcc:.3f}")

        # Confusion matrix
        st.subheader("🔎 Confusion Matrix")
        cm = confusion_matrix(y, preds)
        st.write(cm)

        # Classification report
        st.subheader("📄 Classification Report")
        st.text(classification_report(y, preds))

else:
    st.info("Please upload a CSV file to proceed.")
