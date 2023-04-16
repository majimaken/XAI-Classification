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

# --------------------------------------------------------------

# Define app layout
st.set_page_config(page_title='Classification of New Data', page_icon='⛷️')
st.title('Explainable AI App')

# Sidebar with Inputs
st.sidebar.header("Input for Classification")
age = st.sidebar.slider("Age", int(18), 
                                int(100), 
                                int(18), key="1")
job = st.sidebar.selectbox("Job", list(set(dat["job"])), key="2")
marital = st.sidebar.selectbox("Marital", list(set(dat["marital"])), key="3")
education = st.sidebar.selectbox("Education", list(set(dat["education"])), key="4")
default = st.sidebar.selectbox("Default", list(set(dat["default"])), key="5")
balance = st.sidebar.slider("Balance", int(-10000), 
                                int(100000), 
                                int(0), key="6")
housing = st.sidebar.selectbox("housing", list(set(dat["housing"])), key="7")
loan = st.sidebar.selectbox("loan", list(set(dat["loan"])), key="8")
contact = st.sidebar.selectbox("Contact", list(set(dat["contact"])), key="9")
day = st.sidebar.selectbox("Day", list(set(dat["day"])), key="10")
month = st.sidebar.selectbox("Month", list(set(dat["month"])), key="11")
duration = st.sidebar.slider("Duration", int(dat["duration"].min()), 
                                int(5000), 
                                int(0), key="12")
campaign = st.sidebar.slider("Campaign", int(dat["campaign"].min()), 
                                int(70), 
                                int(0), key = "13")           
pdays = st.sidebar.slider("Days since client was last contacted", int(dat["pdays"].min()), 
                                int(1000), 
                                int(0), key = "14")    
previous = st.sidebar.slider("Number of contacts before this campaign", int(dat["previous"].min()), 
                                int(100), 
                                int(0), key = "15")                                  
poutcome = st.sidebar.selectbox("Outcome of the previous marketing campaign", list(set(dat["poutcome"])), key = "16")

# Build new dataframe for prediction
data = [age, job, marital, education, default, balance, housing, loan, 
        contact, day, month, duration, campaign, pdays, previous, poutcome]

new_df = pd.DataFrame([data],
                      columns = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", 
                                "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"])
                                
st.subheader("Data Used for Classification")
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)
st.table(data = new_df)

# Workaround LabelEncoder()
# Append new_df to X_test in order to get LabelEncoder() to work
new_df_append = df.append(new_df, ignore_index = True)

# Encode columns with object / categorical variables
# 1) Find all columns with categorical / object and assign to list
categorical_cols = new_df_append.select_dtypes(include="object").columns.tolist()    

# 2) Create df_encoded
new_df_encoded = new_df_append.copy()

# 3) Create new dataframe with preprocessed categorical variables
# encoder = LabelEncoder()

for col in categorical_cols:
    new_df_encoded[col] = encoder.fit_transform(new_df_append[col])
    
# 4) Convert all yes/no to binary
new_df_encoded.replace({'yes': 1, 'no': 0}, inplace=True)      

# 5) Select last row of new_df_encoded and drop y-column
new_df_encoded = new_df_encoded.tail(1)
new_df_encoded = new_df_encoded.drop("y", axis = 1)

# Show Encoded Dataset
st.subheader("Preprocessed Data Used for Classification")
# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)
st.table(data = new_df_encoded) 

# # Make prediction
# example = pd.DataFrame(inputs, index=[0])
prediction = xgb_model.predict(new_df_encoded)
probability = xgb_model.predict_proba(new_df_encoded)

# # # Display prediction and probability
st.subheader('Prediction')
st.write('NOT interested in Term Deposit' if prediction == 0 else 'Interested in Term Deposit')
st.subheader('Probabilities')
st.write("Probabilities for not being interested (0) and being interested (1)")
st.write(probability)






