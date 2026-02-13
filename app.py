import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

st.set_page_config(page_title="Adult Income Prediction", layout="centered")

st.title("Adult Income Classification App")
st.write("Upload a test dataset (CSV) to evaluate model performance.")

# -------------------- Model Selection --------------------
model_option = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "Naive Bayes", "Random Forest", "XGBoost"]
)

# -------------------- Load Models --------------------
model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

model = joblib.load(model_files[model_option])
scaler = joblib.load("model/scaler.pkl")
columns = joblib.load("model/columns.pkl")

# -------------------- Upload CSV --------------------
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "income" not in df.columns:
        st.error("Dataset must contain 'income' column.")
        st.stop()

    y = df["income"]
    X = df.drop("income", axis=1)

    # -------------------- Preprocessing --------------------
    X = pd.get_dummies(X)
    X = X.reindex(columns=columns, fill_value=0)
    X_scaled = scaler.transform(X)

    # -------------------- Prediction --------------------
    preds = model.predict(X_scaled)

    # -------------------- Metrics --------------------
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    mcc = matthews_corrcoef(y, preds)

    try:
        prob = model.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y, prob)
        auc_value = f"{auc:.3f}"
    except:
        auc_value = "Not Available"

    # -------------------- Display Metrics --------------------
    st.subheader("Evaluation Metrics")

    col1, col2 = st.columns(2)

    col1.metric("Accuracy", f"{acc:.3f}")
    col1.metric("Precision", f"{prec:.3f}")
    col1.metric("Recall", f"{rec:.3f}")

    col2.metric("F1 Score", f"{f1:.3f}")
    col2.metric("AUC", auc_value)
    col2.metric("MCC", f"{mcc:.3f}")

    # -------------------- Confusion Matrix --------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, preds)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # -------------------- Prediction Summary --------------------
    st.subheader("Prediction Preview")

    preview_df = pd.DataFrame({"Actual": y, "Predicted": preds})
    st.dataframe(preview_df.head(20))
