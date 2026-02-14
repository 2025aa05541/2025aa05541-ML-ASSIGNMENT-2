# Adult Income Classification – ML Assignment 2

## a. Problem Statement

The objective of this project is to build multiple machine learning classification models 
to predict whether an individual earns more than $50K per year based on demographic 
and employment-related attributes. 

This is a binary classification problem where the target variable is income category:
<=50K,>50K
The goal is to compare different classification algorithms using multiple evaluation metrics 
and deploy the best-performing models using Streamlit.


## b. Dataset Description

Dataset Name: Adult Census Income Dataset  
Source: Kaggle (UCI Machine Learning Repository)

Dataset Details:
- Number of Instances: 48,842
- Number of Input Features: 14
- Target Variable: Income (<=50K or >50K)
- Problem Type: Binary Classification

Input Features:
1. age  
2. workclass  
3. fnlwgt  
4. education  
5. education-num  
6. marital-status  
7. occupation  
8. relationship  
9. race  
10. sex  
11. capital-gain  
12. capital-loss  
13. hours-per-week  
14. native-country  

The dataset contains both categorical and numerical features. 
Categorical variables were encoded using Label Encoding during preprocessing.

The dataset satisfies the assignment requirement of:
- Minimum 12 features ✔
- Minimum 500 instances ✔


## c. Models Used and Evaluation

The following six classification models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

### Evaluation Metrics Used:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)


## Comparison Table

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression      | 0.8045   | 0.8115 | 0.6886    | 0.3906 | 0.4985 | 0.4125 |
| Decision Tree            | 0.7998   | 0.7362 | 0.5949    | 0.6100 | 0.6024 | 0.4687 |
| kNN                      | 0.7618   | 0.6529 | 0.5365    | 0.3087 | 0.3919 | 0.2721 |
| Naive Bayes              | 0.7824   | 0.8155 | 0.6410    | 0.2833 | 0.3930 | 0.3190 |
| Random Forest (Ensemble) | 0.8502   | 0.9089 | 0.7854    | 0.5467 | 0.6447 | 0.5681 |
| XGBoost (Ensemble)       | 0.8603   | 0.9206 | 0.7541    | 0.6500 | 0.6982 | 0.6108 |



## Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | Achieved 0.8045 accuracy with moderate AUC (0.8115). Precision is high (0.6886) but recall is relatively low (0.3906), indicating difficulty in correctly identifying >50K cases.            |
| Decision Tree       | Accuracy (0.7998) is slightly lower than Logistic Regression, but recall (0.61) is higher, meaning it detects more high-income individuals. However, AUC (0.7362) is lower.                  |
| kNN                 | Shows lowest overall performance with accuracy 0.7618 and MCC 0.2721. Particularly weak recall (0.3087) suggests poor identification of >50K class.                                          |
| Naive Bayes         | Provides reasonable AUC (0.8155) but very low recall (0.2833), meaning it struggles to classify high-income cases correctly.                                                                 |
| Random Forest       | Strong ensemble performance with accuracy 0.8502 and AUC 0.9089. Balanced precision (0.7854) and recall (0.5467) improve overall F1 score.                                                   |
| XGBoost             | Best performing model with highest accuracy (0.8603), AUC (0.9206), and MCC (0.6108). Recall (0.65) is also the highest among all models, making it the most balanced and robust classifier. |



## Deployment

The application is deployed using Streamlit Community Cloud.
https://2025aa05541-ml-assignment-2-bzqc7eqepvhwqtyeytjvgz.streamlit.app/
Features of the App:
- CSV Test Data Upload
- Model Selection Dropdown
- Display of Evaluation Metrics
- Confusion Matrix Visualization
