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
st.write("Upload test CSV file to evaluate selected model.")

# ---------------- LOAD PREPROCESSING FILES ----------------
scaler = joblib.load("model/scaler.pkl")
columns = joblib.load("model/columns.pkl")

# ---------------- SAFE MODEL LOADING ----------------
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        return None

models = {
    "Logistic Regression": load_model("model/logistic_regression.pkl"),
    "Decision Tree": load_model("model/decision_tree.pkl"),
    "Naive Bayes": load_model("model/naive_bayes.pkl"),
    "Random Forest": load_model("model/random_forest.pkl"),
    "XGBoost": load_model("model/xgboost.pkl"),
    "kNN": None  # will train live
}

# ---------------- MODEL SELECTION ----------------
model_name = st.selectbox("Select ML Model", list(models.keys()))

# ---------------- DATASET UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # -------- CLEAN COLUMN NAMES --------
    df.columns = df.columns.astype(str)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(".", "", regex=False)

    # -------- TARGET DETECTION --------
    target_col = df.columns[-1]

   y = df[target_col].astype(str)
   # clean spaces and dots
   y = y.str.strip()
   y = y.str.replace(".", "", regex=False)

   # normalize text
   y = y.str.replace(" ", "")

   # map labels
   y = y.map({"<=50K":0, ">50K":1})

  # remove rows that failed mapping
  valid_idx = y.notna()
  X = X[valid_idx]
  y = y[valid_idx]

    # -------- ONE HOT ENCODING --------
    X = pd.get_dummies(X)

    # -------- ALIGN COLUMNS WITH TRAINING --------
    for col in columns:
        if col not in X:
            X[col] = 0

    X = X[columns]

    # -------- SCALING --------
    X_scaled = scaler.transform(X)

    # -------- MODEL SELECTION --------
    if model_name == "kNN":
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_scaled, y)
    else:
        model = models[model_name]

    # -------- PREDICTIONS --------
    preds = model.predict(X_scaled)

    try:
        probs = model.predict_proba(X_scaled)[:,1]
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



