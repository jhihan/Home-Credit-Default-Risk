# Home-Credit-Default-Risk
This project is from the kaggle competition https://www.kaggle.com/c/home-credit-default-risk/overview and is a good practice to deal with large relational datasets. The feature engineering process with large relational datasets is error-prone and cumbersome, so the automatic feature engineering library tool "Featuretools" ( https://docs.featuretools.com/en/stable/  ) is used in this project in order to save our life. Before the feature engineering, we need to do some data expolatory analysis to understand the data. This project is seperated into two files:
1. Home_credit_default_risk_app.ipynb :The data expolatory analysis and the machine learning model building based on the main table.
2. Home_credit_default_risk.ipynb : Building the machine learning model based on the automatic feature engineering with "featuretools" and the feature selections process.
3. Home_credit_default_risk_more_feature_engineering.ipynb: Building the machine learning model based on the both automatic and manual feature engineering with "featuretools", and the feature selections process.
## Exploratory data analysis
## Memory reduction
With data size increasing, sometimes we are not able to deal with the data in our disk. Downcasting the data types to the suitable subtypes is an efficient way to reduce the memory.

float: float64, float32, float16

int: int64, int32, int16

[0,1]=> bool

object => category (please know that np.nan is not in categories )
## Feature engineering manually
### date/time variables: 
date/time variables have periodicity. In order to keep this kind of property into consideration, we must take the cos and sin components of these variable.

### Convert the ordinal variables to the numerical variables
Ordinal variables and the categorical variables are very similar, but ordinal variableshave intrinsic order. Some non-numerical variables which might have intrinsic order should be carefully checked. They might be better converted as numerical features rather than categorical variables. For example, the education-related feature "NAME_EDUCATION_TYPE" is a candidate we should check. In order to make sure whether there is significant difference between different values of the feature, we need to estimate the standard deviation of the sample ratio
<img src="https://render.githubusercontent.com/render/math?math=\sigma=\sqrt{\frac{p(1-p)}{N}}">  in order to get the confidence interval (95% confidence level means the interval <img src="https://render.githubusercontent.com/render/math?math=[p-2\sigma,p \+ 2\sigma]">).

### Domain knowledge
We can also create new features based on some domain knowledge. when reading the HomeCredit_columns_descriptions.csv carefully, we can find that some features like DAYS_EMPLOYED (total days of being employed), AMT_INCOME_TOTAL (total income of the applicant per annum), AMT_ANNUITY (the annuity of each credit loan) can be used to create some interesting features like: INCOME_PER_DAYS_EMPLOYED = AMT_INCOME_TOTAL/DAYS_EMPLOYED.

### Treating KNN as feature engineering
We use K-Nearest Neighbors (KNN) to add a "local knowledge" feature. To achieve this purpose, we can choose some features which present different distribution between different class label and then estimate the predicted class probability. Then then this predicted class probability is used as a new feature for downstream modeling. 

Key points: 

1.The choice of features for the KNN relies on the detailed exploratory data analysis.

2. Multicollinearity is not a issue here because the information being incorporated into the second-stage model
is highly local. This is an additional information rather than redundant information.

More discussion about KNN as feature engineering: P.386 from Bruce, Peter, and Andrew Bruce. 2017. Practical Statistics for Data Scientists: 50 Essential Concepts. O’Reilly Media, Inc.

## Feature engineering with Featuretools
Featuretools automatically creates features from relational datasets. This tool use the cencept Deep Feature Synthesis for automated feature engineering. The procedure is as followed:

1. we load the dataframes as entities with the given primary key

2. we define the relationship between different entity

3. We choose the agg primitives function and the depth of a deep feature, which is the number of primitives required to make the feature.
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

## Comparison between different data preprocessing and feature engineerings:

In the tree-based model, the feature importance (feature_importances_) is calculated as the decrease in node impurity weighted by the probability of reaching that node. We can check how important the additional features which are created manually. There are two features we created manually in the top 15 ranking: 'EDUCATION_VALUE' ranks 6 and 'CREDIT_PER_ANNUITY' ranks 11.

We compare the ROC AUC sore of the submitted prediction between different data preprocessing approaches. Apparently, the additional manual feature engineering makes our ROC AUC score about 0.01 better in both private and public score.

|   | Private Score | Public score |  
|---|---|---|
| Only automated feature engineering   | 0.76530  | 0.76449 | 
| Both automated and manual feature engineering  |  0.77566 | 0.77547 |

## Reference
1. https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics
2. https://github.com/sunny1297/Risk-Analytics
3. How to handle BigData Files on Low Memory?
https://towardsdatascience.com/how-to-learn-from-bigdata-files-on-low-memory-incremental-learning-d377282d38ff
4. Optimize the Pandas Dataframe memory consuming for low environment
https://medium.com/@alielagrebi/optimize-the-pandas-dataframe-memory-consuming-for-low-environment-24aa74cf9413
5. Collinearity: a review of methods to deal with it and a simulation study evaluating their performance
https://onlinelibrary.wiley.com/doi/10.1111/j.1600-0587.2012.07348.x
6. Bruce, Peter, and Andrew Bruce. 2017. Practical Statistics for Data Scientists: 50 Essential Concepts. O’Reilly Media, Inc
