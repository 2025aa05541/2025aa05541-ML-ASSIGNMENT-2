import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Adult Income Classification App")

st.write("Upload test dataset and select model for prediction.")

# Load scaler
scaler = joblib.load("model/scaler.pkl")

# Model selection
model_option = st.selectbox(
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

# Load selected model
def load_model(option):
    if option == "Logistic Regression":
        return joblib.load("model/logistic.pkl")
    elif option == "Decision Tree":
        return joblib.load("model/decision_tree.pkl")
    elif option == "KNN":
        return joblib.load("model/knn.pkl")
    elif option == "Naive Bayes":
        return joblib.load("model/naive_bayes.pkl")
    elif option == "Random Forest":
        return joblib.load("model/random_forest.pkl")
    elif option == "XGBoost":
        return joblib.load("model/xgboost.pkl")

model = load_model(model_option)

# File upload
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Dataset Preview:")
    st.dataframe(data.head())

    # Handle missing values
    data.replace(" ?", np.nan, inplace=True)
    data.dropna(inplace=True)

    # Encoding
    data = pd.get_dummies(data, drop_first=True)

    # Target detection
    target_column = [col for col in data.columns if '>50K' in col or 'income' in col.lower()]
    
    if len(target_column) == 0:
        st.error("Target column not found in uploaded file.")
    else:
        target_column = target_column[-1]

        X = data.drop(target_column, axis=1)
        y = data[target_column]

        # Scaling
        X = scaler.transform(X)

        # Prediction
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:,1]

        # Metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)
        auc = roc_auc_score(y, y_prob)

        st.subheader("Evaluation Metrics")
        st.write("Accuracy:", accuracy)
        st.write("AUC:", auc)
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("F1 Score:", f1)
        st.write("MCC:", mcc)

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)
