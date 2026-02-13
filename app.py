import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

st.title("Adult Income Classification App")

# Model selection
model_name = st.selectbox("Select Model", [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
])

uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    X = data.drop("income", axis=1)
    y = data["income"]
    
    model = joblib.load(f"model/{model_name}.pkl")
    
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]
    
    st.subheader("Evaluation Metrics")
    st.write("Accuracy:", accuracy_score(y, y_pred))
    st.write("AUC:", roc_auc_score(y, y_prob))
    st.write("Precision:", precision_score(y, y_pred))
    st.write("Recall:", recall_score(y, y_pred))
    st.write("F1 Score:", f1_score(y, y_pred))
    st.write("MCC:", matthews_corrcoef(y, y_pred))
    
    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))
