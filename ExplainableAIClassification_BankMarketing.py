# Bank Marketing XAI App

# Load Libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataprep.eda import plot, plot_correlation, create_report, plot_missing # Dataprep is an automated Data Exploration package
# import sweetviz as sv # SweetViz is an automated Data Exploration package
# import warnings # Control how often we want to see warnings with warnings.filterwarnings("ignore") or action = "once"

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score,accuracy_score,ConfusionMatrixDisplay
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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
xgb = XGBClassifier(scale_pos_weight = 6.5, 
                    eval_metric = "auc",
                    learning_rate = 0.3,
                    max_depth = 5,
                    n_estimators = 40)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)



# --------------------------------------------------------------

# Define app layout
st.set_page_config(page_title='Explaining XGBoost Classification', page_icon='🤖')
st.title('Explainable AI App')
st.header('Model predictions')

# Show Data Frame
st.title("Overview of Data Set")
url= "https://raw.githubusercontent.com/majimaken/XAI-Classification/main/bank-full.csv"
dat = pd.read_csv(url, sep = ";")
st.dataframe(data = dat)

# Display Confusion Matrix
st.subheader("Confusion Matrix")
st.markdown("""
A confusion matrix is a table that visualizes the performance of a machine learning
model by comparing the predicted and actual values of a classification problem.
It contains information about the number of true positives, true negatives,
false positives, and false negatives, which are used to calculate various 
evaluation metrics such as accuracy, precision, recall, and F1 score.
""")
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm).plot(values_format = "", cmap = "Greens")
plt.title("Confusion Matrix")
plt.text(-0.4, 0.2, "correctly classified as 0", fontsize = 9, color = "w")
plt.text(-0.4, 1.2, "mistakenly classified as 0", fontsize = 9)
plt.text(0.6, 0.2, "mistakenly classified as 1", fontsize = 9)
plt.text(0.6, 1.2, "correctly classified as 1", fontsize = 9)
plt.show()
st.pyplot()

# Display the SHAP values for the example
st.subheader('Feature Importance')

# Feature Importances
importances = xgb.feature_importances_
feature_names = X.columns

# Create a dataframe of feature importances
df_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})

# Sort the dataframe by importance
df_importances = df_importances.sort_values('importance', ascending=True)

# Create a horizontal bar chart of feature importances
plt.barh(df_importances['feature'], df_importances['importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
st.pyplot(plt)


# Sidebar
st.sidebar.header("Input for Classification")
age = st.sidebar.slider("Age", float(dat["age"].min()), 
                                float(dat["age"].max()), 
                                float(dat["age"].mean()))
job = st.sidebar.selectbox("Job", list(set(dat["job"])))
marital = st.sidebar.selectbox("Marital", list(set(dat["marital"])))



# Get user input for predictor variables
features = df_preprocessed.drop("y", axis = 1).columns.values

inputs = {}
for feature in features:
    inputs[feature] = st.slider(feature, 
                                float(df[feature].min()), 
                                float(df[feature].max()), 
                                float(df[feature].mean()))

# Make prediction
example = pd.DataFrame(inputs, index=[0])
prediction = xgb.predict(example)[0]
probability = xgb.predict_proba(example)[0][1]

# Display prediction and probability
st.subheader('Prediction')
st.write('Not interested in Term Deposit' if prediction == 0 else 'Interested in Term Deposit')
st.subheader('Probability')
st.write(probability)









