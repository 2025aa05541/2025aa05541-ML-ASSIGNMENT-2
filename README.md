**a. Problem Statement**
       The objective of this project is to build and compare multiple machine learning classification models to predict whether an individual's annual income exceeds $50K based on demographic and employment-related attributes.
The goal is to evaluate different algorithms using multiple performance metrics and deploy the best-performing model as an interactive Streamlit web application.

**b. Dataset description** 
| Attribute           | Details                         |
| ------------------- | ------------------------------- |
| Dataset Name        | UCI Adult Income Dataset        |
| Source              | UCI Machine Learning Repository |
| Problem Type        | Binary Classification           |
| Target Variable     | Income (>50K / <=50K)           |
| Number of Instances | ~48,000                         |
| Number of Features  | 14 (before encoding)            |
| Data Type           | Mixed (Categorical + Numerical) |
The dataset contains demographic and employment attributes such as age, education, occupation, hours-per-week, marital status, etc.

**c. Models Used and Evaluation Metrics**

**Model Comparison Table**

| ML Model Name            | Accuracy | AUC      | Precision | Recall   | F1       | MCC      |
| ------------------------ | -------- | -------- | --------- | -------- | -------- | -------- |
| Logistic Regression      | 0.942257 | 0.944585 | 0.730392  | 0.403794 | 0.520070 | 0.516625 |
| Decision Tree            | 0.917900 | 0.727766 | 0.471939  | 0.501355 | 0.486202 | 0.441871 |
| kNN                      | 0.925669 | 0.823088 | 0.535545  | 0.306233 | 0.389655 | 0.368899 |
| Naive Bayes              | 0.540997 | 0.752480 | 0.142604  | 0.982385 | 0.249055 | 0.260627 |
| Random Forest (Ensemble) | 0.941522 | 0.925896 | 0.717026  | 0.405149 | 0.517749 | 0.511853 |
| XGBoost (Ensemble)       | 0.947192 | 0.954096 | 0.755991  | 0.470190 | 0.579783 | 0.571047 |


**Observations on Model Performance**

| ML Model Name       | Observation about Model Performance                                                                 |
| ------------------- | --------------------------------------------------------------------------------------------------- |
| Logistic Regression | Achieved high AUC and balanced performance after handling class imbalance. Good baseline model.     |
| Decision Tree       | Moderate performance but lower AUC compared to ensemble models. Slight overfitting tendency.        |
| kNN                 | Reasonable accuracy but lower recall, indicating difficulty detecting high-income class.            |
| Naive Bayes         | Extremely high recall but very low precision, meaning many false positives.                         |
| Random Forest       | Strong balanced performance across all metrics. More stable than single tree.                       |
| XGBoost             | Best overall performance with highest AUC, F1-score, and MCC. Most reliable model for this dataset. |


**Deployment**
**🔗 Live Streamlit App Link:** https://2025aa05541-ml-assignment-2-bzqc7eqepvhwqtyeytjvgz.streamlit.app/

