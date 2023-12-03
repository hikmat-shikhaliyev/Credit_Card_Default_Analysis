# Credit_Card_Default_Analysis
This repository contains a machine learning project focused on predicting the 'default' status. The project involves the use of diverse machine learning models, hyperparameter tuning, and univariate analysis.
### Introduction
This project aims to predict the 'default' status using machine learning models. The README provides an overview of the models used, hyperparameter tuning, and the analysis conducted.
### Dataset
The dataset includes features for predicting the 'default' status. Preprocessing steps have been applied to prepare the data for model training.
### Preprocessing
Handling Missing Values
#### Imputation: 
Addressed missing values in numerical features using appropriate imputation techniques like mean or median.
#### Categorical Imputation: 
Filled missing categorical values with the mode, ensuring data integrity and consistency.
#### Feature Engineering
#### Weight of Evidence (WOE) Transformation
#### Categorical Variables Transformation: 
Implemented Weight of Evidence (WOE) transformation on categorical variables specifically for the LogisticRegression model.
#### Handling Outliers
#### Outlier Detection: 
Identified and addressed outliers using statistical methods or domain knowledge to prevent them from adversely affecting the model.
#### Scaling
#### Numerical Feature Scaling: 
Applied feature scaling to numerical features, ensuring a consistent scale for improved model convergence.
Encoding
#### One-Hot Encoding: 
Employed one-hot encoding for categorical variables, converting them into a format suitable for machine learning models.
# Models
#### CatBoost Classifier
Trained and evaluated a CatBoost classifier with default settings.
Conducted hyperparameter tuning using RandomizedSearchCV.
Explored univariate analysis to understand the impact of individual variables.
#### XGBoost Classifier
Trained and evaluated an XGBoost classifier with default settings.
Performed hyperparameter tuning using RandomizedSearchCV.
#### LightGBM Classifier
Trained and evaluated a LightGBM classifier with default settings.
Conducted hyperparameter tuning using RandomizedSearchCV.
#### Random Forest Classifier
Trained and evaluated a Random Forest classifier with default settings.
Explored hyperparameter tuning using RandomizedSearchCV.
#### Stacking Classifier
Built a stacking classifier using CatBoost, XGBoost, LightGBM, and Random Forest as base classifiers.
Used CatBoost as the meta-classifier.
#### Support Vector Classifier (SVC)
Trained and evaluated a Support Vector Classifier (SVC) with default settings.
Explored hyperparameter tuning using RandomizedSearchCV.
#### Hyperparameter Tuning
CatBoost Hyperparameter Tuning
Outlined the process of hyperparameter tuning for the CatBoost model.

XGBoost Hyperparameter Tuning
Explained the process of hyperparameter tuning for the XGBoost model.

LightGBM Hyperparameter Tuning
Described the process of hyperparameter tuning for the LightGBM model.

SVC Hyperparameter Tuning
Detailed the process of hyperparameter tuning for the Support Vector Classifier (SVC).

### Univariate Analysis
Explored the impact of individual variables on predictive performance through univariate analysis.

CatBoost Model with Categorical Columns
Demonstrated the use of CatBoost with categorical columns and evaluated its performance.

### Results
Summarized the results of each model, including Gini scores, ROC curves, and relevant metrics.


