# Home-Credit-Default-Risk
This project is from the kaggle competition https://www.kaggle.com/c/home-credit-default-risk/overview and is a good practice to deal with large relational datasets. The feature engineering process with large relational datasets is error-prone and cumbersome, so the automatic feature engineering library tool "Featuretools" ( https://docs.featuretools.com/en/stable/  ) is used in this project in order to save our life. Before the feature engineering, we need to do some data expolatory analysis to understand the data. This project is seperated into two files:
1. Home_credit_default_risk_app.ipynb :The data expolatory analysis and the machine learning model building based on the main table.
2. Home_credit_default_risk.ipynb : Building the machine learning model based on the automatic feature engineering with "featuretools" and the feature selections process.
## Data expolatory analysis
## Memory reduction
## Feature engineering with Featuretools
## Feature selection
### Removing unnecessary features
### Removing columns with too many missing values
### Rreducing collinearity
## Encoding Categorical Data
## Building model with XGboost
