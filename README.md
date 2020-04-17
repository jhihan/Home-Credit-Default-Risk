# Home-Credit-Default-Risk
This project is from the kaggle competition https://www.kaggle.com/c/home-credit-default-risk/overview and is a good practice to deal with large relational datasets. The feature engineering process with large relational datasets is error-prone and cumbersome, so the automatic feature engineering library tool "Featuretools" ( https://docs.featuretools.com/en/stable/  ) is used in this project in order to save our life. Before the feature engineering, we need to do some data expolatory analysis to understand the data. This project is seperated into two files:
1. Home_credit_default_risk_app.ipynb :The data expolatory analysis and the machine learning model building based on the main table.
2. Home_credit_default_risk.ipynb : Building the machine learning model based on the automatic feature engineering with "featuretools" and the feature selections process.
## Exploratory data analysis
Featuretools automatically creates features from relational datasets. This tool use the cencept Deep Feature Synthesis for automated feature engineering. The procedure is as followed:

1. we load the dataframes as entities with the given primary key

2. we define the relationship between different entity

3. We choose the agg primitives function and the depth of a deep feature, which is the number of primitives required to make the feature.

## Memory reduction
With data size increasing, sometimes we are not able to deal with the data in our disk. Downcasting the data types to the suitable subtypes is an efficient way to reduce the memory.

float: float64, float32, float16

int: int64, int32, int16

[0,1]=> bool

object => category (please know that np.nan is not in categories )
## Feature engineering with Featuretools
## Feature selection
The feature matrix calculated from the last step is very large (will be larger after encoding categorical data) and contains lots of missing values. Without feature selection, the training process of ML models would be very slow. Some of the redundant features might even decrease the performance of the model. Therefore, getting rid of some unnecessary or redundant information is an important step.
### Removing unnecessary features
After the data engineering from Featuretools, we will also get some column variables which are the function of keys. Because keys are data items that exclusively identify a record and are not the features which are useful in the model.
### Removing columns with too many missing values
The columns with too many missing values contain too less information. Imputation of these missing data is not a good idea and we should set up the minimum threshold of percentage of missing values for removing. In addition, We must pay attention to a special situation: there are two unique values and one of them is nan, such as "approved" and nan. In this kind of the special case. The missing value DOES contain information.
### Rreducing collinearity
Collinear features lead to decreased generalization performance on the test set due to the high variance and the accessibility to some relative importance of variables. In order to solve this problem, only one of the collinear feature is preserved and others are removed. In order to achieve this purpose, the correlation matrix must be calculated first. Then we traverse across the strickly upper triangular part of correlation matrix to remove a highly correlated variable (here threshold = 0.9) in the column of the matrix.
## Encoding categorical data
In order to convert all the input of the machine learning data into numerical datatype, we must do encoder for categorical variables. There are two ways: 

1. label encoder: convert each value of a categorical variable into single integer. However, the model will misunderstand the data to be in some kind of order. This encoder should be only used if the original values of the categorical variables have intrinsic order or the categorical variables are binary variables.

2. one-hot encoder: convert each categorical value into a new binary variable. However, this operation will increase the number of feature space.

Therefore, We use label encoder for categorical variables with 2 unique categories (or 1 categories + np.nan) and one-hot Encoder for categorical variables with more than 2 unique categories. When doing label encoder with np.nan, we must be careful because the datatype "category" doesn't recognize np.nan. In this case, we should add a new category such as "NAN" into the categories-list  and fill the missing value with "NAN".
## Building model with XGBoost
There are still many columns with missing values. One way to fill the missing values is impute these value with mean, average or some values which we can infer from the domain knowledge. However, doing missing data imputation manually in the large datasets is tedious. Therefore, choosing a library which automatically can handle missing data is a favorable approach. Two of the famous libraries which fulfill our need are XGBoost and LightGBM. They are both gradient boosting framework and are popular in kaggle competitions. Here we use XGBoost tree classifier to build up the model.
## Reference
1. https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics
2. https://github.com/sunny1297/Risk-Analytics
3. How to handle BigData Files on Low Memory?
https://towardsdatascience.com/how-to-learn-from-bigdata-files-on-low-memory-incremental-learning-d377282d38ff
4. Optimize the Pandas Dataframe memory consuming for low environment
https://medium.com/@alielagrebi/optimize-the-pandas-dataframe-memory-consuming-for-low-environment-24aa74cf9413
4. Collinearity: a review of methods to deal with it and a simulation study evaluating their performance
https://onlinelibrary.wiley.com/doi/10.1111/j.1600-0587.2012.07348.x
