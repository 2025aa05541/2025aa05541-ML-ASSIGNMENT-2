import streamlit as st
import pandas as pd
import joblib
import os
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



st.markdown("## 📈 Training Data Model Comparison")

comparison_data = {
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy": [0.8178, 0.8034, 0.7618, 0.7824, 0.8478, 0.8667],
    "AUC": [0.8447, 0.7418, 0.6529, 0.8155, 0.9063, 0.9214],
    "Precision": [0.7182, 0.6017, 0.5365, 0.6410, 0.7916, 0.7766],
    "Recall": [0.4400, 0.6193, 0.3087, 0.2833, 0.5267, 0.6513],
    "F1 Score": [0.5457, 0.6104, 0.3919, 0.3930, 0.6325, 0.7085],
    "MCC": [0.4605, 0.4791, 0.2721, 0.3190, 0.5593, 0.6270]
}

comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True)

st.markdown("---")


model_name = st.selectbox(
    "Select Model for Testing",
    comparison_data["Model"]
)

uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])



st.markdown("## 📊 Evaluation Metrics (Test Data)")

acc_placeholder = st.empty()
auc_placeholder = st.empty()
prec_placeholder = st.empty()
rec_placeholder = st.empty()
f1_placeholder = st.empty()
mcc_placeholder = st.empty()

# Default template values
acc_placeholder.write("Accuracy: -")
auc_placeholder.write("AUC: -")
prec_placeholder.write("Precision: -")
rec_placeholder.write("Recall: -")
f1_placeholder.write("F1 Score: -")
mcc_placeholder.write("MCC: -")


st.markdown("## 🔢 Confusion Matrix")

cm_placeholder = st.empty()

default_cm = [[0, 0], [0, 0]]

fig, ax = plt.subplots(figsize=(5, 4))

sns.heatmap(
    default_cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["<=50K", ">50K"],
    yticklabels=["<=50K", ">50K"],
    ax=ax
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_title("Confusion Matrix")

cm_placeholder.pyplot(fig)



if uploaded_file is None:
    st.info("Metrics and confusion matrix values will be displayed once the test dataset is uploaded.")



else:
    data = pd.read_csv(uploaded_file)

    X = data.drop("income", axis=1)
    y = data["income"]

    model_files = {
        "Logistic Regression": "Logistic Regression.pkl",
        "Decision Tree": "Decision Tree.pkl",
        "KNN": "KNN.pkl",
        "Naive Bayes": "Naive Bayes.pkl",
        "Random Forest": "Random Forest.pkl",
        "XGBoost": "XGBoost.pkl"
    }

    model_path = os.path.join("model", model_files[model_name])
    model = joblib.load(model_path)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

   
    acc_placeholder.write(f"Accuracy: {round(accuracy_score(y, y_pred), 4)}")
    auc_placeholder.write(f"AUC: {round(roc_auc_score(y, y_prob), 4)}")
    prec_placeholder.write(f"Precision: {round(precision_score(y, y_pred), 4)}")
    rec_placeholder.write(f"Recall: {round(recall_score(y, y_pred), 4)}")
    f1_placeholder.write(f"F1 Score: {round(f1_score(y, y_pred), 4)}")
    mcc_placeholder.write(f"MCC: {round(matthews_corrcoef(y, y_pred), 4)}")

    
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["<=50K", ">50K"],
        yticklabels=["<=50K", ">50K"],
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")

    cm_placeholder.pyplot(fig)
