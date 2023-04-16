# Bank Marketing XAI App

# Load Libraries
import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import xgboost as xgb
from xgboost import XGBClassifier, plot_importance

from sklearn.preprocessing import LabelEncoder

import shap

# Disable Warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load Data
url= "https://raw.githubusercontent.com/majimaken/XAI-Classification/main/bank-full.csv"
dat = pd.read_csv(url, sep = ";")
df = dat

# Function for Preprocessing Dataset
def convert_categorical_to_numerical(df):
    """
    Convert categorical variables into numerical variables using LabelEncoder from scikit-learn
    
    Parameters:
    -----------
    df : pandas dataframe
        The dataframe containing the categorical variables
    
    Returns:
    --------
    df : pandas dataframe
        The dataframe with converted categorical variables
    
    label_encoders : dict
        Dictionary containing the LabelEncoder object for each categorical column
    """
    
    # Convert all yes/no to binary
    df.replace({'yes': 1, 'no': 0}, inplace=True)
    
    # Find all columns with categorical / object and assign to list
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders

# Acquiring preprocessed dataset and labels
df_preprocessed, label_enc = convert_categorical_to_numerical(df)

# Feature Engineering
features = df_preprocessed.drop("y", axis = 1).columns.values
X = df_preprocessed[features]
y = df_preprocessed["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state= 2023)

# Fitting Weighted XGBoost Model
xgb_model = XGBClassifier(scale_pos_weight = 6.5, 
                    eval_metric = "auc",
                    learning_rate = 0.3,
                    max_depth = 5,
                    n_estimators = 40)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)



# --------------------------------------------------------------

# Define app layout
st.set_page_config(page_title='Feature Importance', page_icon='🔝')
st.title('Feature Importance')


st.header("XAI for XGBoost")

st.markdown("""
There are several ways to do XAI (Explainable Artificial Intelligence) with XGBoost, some of the most common methods include:

Feature Importance: 
This method involves calculating the relative importance of each input feature in the XGBoost model. The importance score is based on how much the feature contributes to reducing the loss function. This method is useful for identifying the most significant features that influence the model's output.

Partial Dependence Plot (PDP): 
This method involves analyzing how the predicted probability or output of the model varies with changes in a single input feature while holding all other features constant. PDPs are useful for identifying complex relationships between input features and the model's output.

Shapley Additive Explanations (SHAP): 
SHAP is a popular XAI technique that provides an explanation for each prediction of the model. SHAP values represent the contribution of each feature to the predicted outcome. These values can be used to understand how the model arrived at its prediction and to identify features that have the greatest impact on the model's output.
""")



# Display the SHAP values for the example
st.subheader('Feature Importance')

# Feature Importances
importances = xgb_model.feature_importances_
feature_names = X.columns

# Create a dataframe of feature importances
df_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})

# Sort the dataframe by importance
df_importances = df_importances.sort_values('importance', ascending=True)

# Create a horizontal bar chart of feature importances
plt.barh(df_importances['feature'], df_importances['importance']) #, color = "navy")
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
st.pyplot(plt)



# Explaining the Prediction with SHAP values
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)

st.header("Feature Importance")
plt.title("Feature Importance based on SHAP")
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches = "tight")
st.write("----")











