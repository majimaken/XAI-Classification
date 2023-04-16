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

# Encode columns with object / categorical variables
# 1) Find all columns with categorical / object and assign to list
categorical_cols = df.select_dtypes(include='object').columns.tolist()    
   
# 2) Create df_encoded
df_encoded = df.copy()

# 3) Create new dataframe with preprocessed categorical variables
encoder = LabelEncoder()

for col in categorical_cols:
    df_encoded[col] = encoder.fit_transform(df[col])
    
# 4) Convert all yes/no to binary
df_encoded.replace({'yes': 1, 'no': 0}, inplace=True)

# Feature Engineering
features = df_encoded.drop("y", axis = 1).columns.values
X = df_encoded[features]
y = df_encoded["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state= 2023)

# Fitting Weighted XGBoost Model
xgb_model = XGBClassifier(scale_pos_weight = 6.5, 
                    eval_metric = "auc",
                    learning_rate = 0.3,
                    max_depth = 5,
                    n_estimators = 40)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

###############################################################

# Define app layout
st.set_page_config(page_title='Partial Dependence', page_icon='🤖')
st.title('Partial Dependence')

st.subheader("Partial Dependence Plot")
st.markdown(
'''
Partial dependence plots (PDPs) are a useful tool for visualizing the relationship between a feature and the target variable in a machine learning model. They show how the predicted outcome changes as the feature value varies, while holding all other features constant.

A PDP is created by first selecting a feature of interest, then generating a series of test cases where that feature is varied while all other features are held constant. For each test case, the predicted outcome is recorded, and the average outcome is computed for each unique value of the feature.

The resulting plot shows how the predicted outcome varies as the feature value changes. This can help to identify non-linear relationships between the feature and the target variable, as well as interactions between multiple features.
'''
)

# Dropdown menu
selected_feature = st.selectbox('Select a feature to visualize:', X_train.columns)

# Calculate partial dependence values for the selected feature
feature_index = X.columns.get_loc(selected_feature)

shap.partial_dependence_plot(
    ind = feature_index, 
    model = xgb_model.predict, 
    data = X,
    ice=False,
    model_expected_value=True, feature_expected_value=True
)
st.pyplot()
