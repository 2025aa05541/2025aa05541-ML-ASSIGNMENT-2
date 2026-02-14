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
| Logistic Regression      | 0.8178   | 0.8447 | 0.7182    | 0.4400 | 0.5457 | 0.4605 |
| Decision Tree            | 0.8034   | 0.7418 | 0.6017    | 0.6193 | 0.6104 | 0.4791 |
| kNN                      | 0.7618   | 0.6529 | 0.5365    | 0.3087 | 0.3919 | 0.2721 |
| Naive Bayes              | 0.7824   | 0.8155 | 0.6410    | 0.2833 | 0.3930 | 0.3190 |
| Random Forest (Ensemble) | 0.8478   | 0.9063 | 0.7916    | 0.5267 | 0.6325 | 0.5593 |
| XGBoost (Ensemble)       | 0.8667   | 0.9214 | 0.7766    | 0.6513 | 0.7085 | 0.6270 |



## Model Performance Observations

| ML Model Name            | Observation about model performance                                                                                                                                                                                         |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Achieved accuracy of 0.8178 with strong AUC (0.8447). Precision is high (0.7182), but recall (0.44) indicates moderate difficulty in identifying >50K cases. After scaling, convergence improved and performance increased. |
| Decision Tree            | Shows balanced recall (0.6193), meaning it detects more high-income individuals compared to Logistic Regression. However, AUC (0.7418) is lower, indicating weaker probability discrimination.                              |
| kNN                      | Lowest overall performance with accuracy 0.7618 and MCC 0.2721. Particularly weak recall (0.3087), suggesting poor identification of >50K class. Sensitive to feature scaling and distance metric.                          |
| Naive Bayes              | Provides good AUC (0.8155), indicating decent ranking capability, but very low recall (0.2833). This suggests it struggles to correctly classify high-income individuals due to independence assumption.                    |
| Random Forest (Ensemble) | Strong ensemble performance with high AUC (0.9063) and good precision (0.7916). Balanced improvement in F1 (0.6325) and MCC (0.5593), reducing overfitting compared to Decision Tree.                                       |
| XGBoost (Ensemble)       | Best performing model with highest accuracy (0.8667), AUC (0.9214), and MCC (0.6270). Also achieved highest recall (0.6513), making it the most balanced and robust classifier among all models.                            |



## Deployment

The application is deployed using Streamlit Community Cloud.
https://2025aa05541-ml-assignment-2-bzqc7eqepvhwqtyeytjvgz.streamlit.app/
Features of the App:
- CSV Test Data Upload
- Model Selection Dropdown
- Display of Evaluation Metrics
- Confusion Matrix Visualization
