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

st.title("Adult Income Classification App")

# Model selection dropdown
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

# Upload CSV
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    X = data.drop("income", axis=1)
    y = data["income"]

    # Load selected model
    model = joblib.load(f"model/{model_name}.pkl")

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    st.subheader("Evaluation Metrics")

    st.write("Accuracy:", round(accuracy_score(y, y_pred), 4))
    st.write("AUC:", round(roc_auc_score(y, y_prob), 4))
    st.write("Precision:", round(precision_score(y, y_pred), 4))
    st.write("Recall:", round(recall_score(y, y_pred), 4))
    st.write("F1 Score:", round(f1_score(y, y_pred), 4))
    st.write("MCC:", round(matthews_corrcoef(y, y_pred), 4))

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
