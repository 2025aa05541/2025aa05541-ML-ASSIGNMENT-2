import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

st.set_page_config(page_title="Adult Income Classification", layout="centered")
st.title("Adult Income Classification App")

# Model Selection
model_name = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# File Upload
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

# ================= METRICS =================
st.subheader("Evaluation Metrics")

acc_p = st.empty()
auc_p = st.empty()
prec_p = st.empty()
rec_p = st.empty()
f1_p = st.empty()
mcc_p = st.empty()

# Default metric placeholders
acc_p.write("Accuracy: -")
auc_p.write("AUC: -")
prec_p.write("Precision: -")
rec_p.write("Recall: -")
f1_p.write("F1 Score: -")
mcc_p.write("MCC: -")

# ================= CONFUSION MATRIX =================
st.subheader("Confusion Matrix")

cm_placeholder = st.empty()

# Show EMPTY confusion matrix template initially
fig, ax = plt.subplots(figsize=(5, 4))
default_cm = [[0, 0], [0, 0]]

sns.heatmap(
    default_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["<=50K", ">50K"],
    yticklabels=["<=50K", ">50K"],
    ax=ax
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_title("Confusion Matrix")

cm_placeholder.pyplot(fig)

# ================= AFTER FILE UPLOAD =================
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    X = data.drop("income", axis=1)
    y = data["income"]

    model = joblib.load(f"model/{model_name}.pkl")

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Update metrics
    acc_p.write(f"Accuracy: {round(accuracy_score(y, y_pred), 4)}")
    auc_p.write(f"AUC: {round(roc_auc_score(y, y_prob), 4)}")
    prec_p.write(f"Precision: {round(precision_score(y, y_pred), 4)}")
    rec_p.write(f"Recall: {round(recall_score(y, y_pred), 4)}")
    f1_p.write(f"F1 Score: {round(f1_score(y, y_pred), 4)}")
    mcc_p.write(f"MCC: {round(matthews_corrcoef(y, y_pred), 4)}")

    # Update confusion matrix
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["<=50K", ">50K"],
        yticklabels=["<=50K", ">50K"],
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")

    cm_placeholder.pyplot(fig)
