# Home-Credit-Default-Risk
This project is from the kaggle competition https://www.kaggle.com/c/home-credit-default-risk/overview and is a good practice to deal with large relational datasets. The feature engineering process with large relational datasets is error-prone and cumbersome, so the automatic feature engineering library tool "Featuretools" ( https://docs.featuretools.com/en/stable/  ) is used in this project in order to save our life. Before the feature engineering, we need to do some data expolatory analysis to understand the data. This project is seperated into two files:
1. Home_credit_default_risk_app.ipynb :The data expolatory analysis and the machine learning model building based on the main table.
2. Home_credit_default_risk.ipynb : Building the machine learning model based on the automatic feature engineering with "featuretools" and the feature selections process.
## Data expolatory analysis
## Memory reduction
## Feature engineering with Featuretools
## Feature selection
The feature matrix from the last step is very large (will be larger after encoding categorical data) and contains lots of missing values. Without feature selection, the training process of ML models would be very slow. Some of the redundant features might even decrease the performance of the model. Therefore, getting rid of some unnecessary or redundant information is an important step.
### Removing unnecessary features
After the data engineering from Featuretools, we will also get some column variables which are the function of keys. Because keys are data items that exclusively identify a record and are not the features which are useful in the model.
### Removing columns with too many missing values
The columns with too many missing values contain too less information. Imputation of these missing data is not a good idea and we should set up the minimum threshold of percentage of missing values for removing. In addition, We must pay attention to a special situation: there are two unique values and one of them is nan, such as "approved" and nan. In this kind of the special case. The missing value DOES contain information.
### Rreducing collinearity
Collinear features lead to decreased generalization performance on the test set due to the high variance and the accessibility to some relative importance of variables. In order to solve this problem, only one of the collinear feature is preserved and others are removed. In order to achieve this purpose, the correlation matrix must be calculated first. Then we traverse across the strickly upper triangular part of correlation matrix to remove a highly correlated variable (here threshold = 0.9) in the column of the matrix.
## Encoding categorical data
## Building model with XGboost

Ref:
1. https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics
2. https://github.com/sunny1297/Risk-Analytics
3. Collinearity: a review of methods to deal with it and a simulation study evaluating their performance
https://onlinelibrary.wiley.com/doi/10.1111/j.1600-0587.2012.07348.x
