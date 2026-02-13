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

| ML Model Name            | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
| ------------------------ | -------- | ----- | --------- | ------ | -------- | ----- |
| Logistic Regression      | 0.942    | 0.944 | 0.730     | 0.404  | 0.520    | 0.516 |
| Decision Tree            | 0.918    | 0.730 | 0.472     | 0.505  | 0.488    | 0.444 |
| kNN                      | 0.926    | 0.823 | 0.536     | 0.306  | 0.390    | 0.369 |
| Naive Bayes              | 0.541    | 0.752 | 0.143     | 0.982  | 0.249    | 0.261 |
| Random Forest (Ensemble) | 0.941    | 0.925 | 0.706     | 0.407  | 0.516    | 0.508 |
| XGBoost (Ensemble)       | 0.947    | 0.954 | 0.756     | 0.470  | 0.580    | 0.571 |

**Observations on Model Performance**

| ML Model Name            | Observation about Model Performance                                                                                                |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Achieved high accuracy and AUC, but recall is relatively low due to class imbalance. Performs well for linear decision boundaries. |
| Decision Tree            | Captures non-linear relationships but slightly lower overall performance compared to Logistic Regression.                          |
| kNN                      | Moderate accuracy but lower recall and F1 score. Sensitive to scaling and high-dimensional data.                                   |
| Naive Bayes              | Very high recall but extremely low precision, leading to poor overall accuracy. Predicts majority of cases as positive class.      |
| Random Forest (Ensemble) | Improved performance compared to single Decision Tree due to ensemble learning. Balanced precision and recall.                     |
| XGBoost (Ensemble)       | Best performing model with highest Accuracy, AUC, F1 Score, and MCC. Effectively handles complex patterns and class imbalance.     |

**Deployment**
**🔗 Live Streamlit App Link:** https://2025aa05541-ml-assignment-2-bzqc7eqepvhwqtyeytjvgz.streamlit.app/
