# Bank Marketing XAI App

# Load Libraries
from PIL import Image
from io import BytesIO
import requests
import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score,accuracy_score,ConfusionMatrixDisplay

from sklearn.tree import DecisionTreeClassifier, plot_tree
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance

from sklearn.preprocessing import LabelEncoder


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
st.set_page_config(page_title='Model Evaluation', page_icon='🔎')
st.title('Model Evaluation')



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
cmd = ConfusionMatrixDisplay(cm).plot(values_format = "", cmap = "YlGnBu")
plt.title("Confusion Matrix")
plt.text(-0.4, 0.2, "correctly classified as 0", fontsize = 9, color = "w")
plt.text(-0.4, 1.2, "mistakenly classified as 0", fontsize = 9) #, color = "b")
plt.text(0.6, 0.2, "mistakenly classified as 1", fontsize = 9) #, color = "b")
plt.text(0.6, 1.2, "correctly classified as 1", fontsize = 9) #, color = "b")
plt.grid(False)
plt.show()
st.pyplot()


# Display the SHAP values for the example
st.subheader('ROC Curve')

# Predict the probabilities of the positive class
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred_proba)

# Get FPR, TPR and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Plot the ROC curve
plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc_score))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
st.pyplot()




# Show chosen Hyperparameters
st.subheader("Hyperparameters")

st.markdown('''
scale_pos_weight = 6.5, 
eval_metric = "auc",
learning_rate = 0.3,
max_depth = 5,
n_estimators = 40
''')





# Show decision tree stump
st.subheader("Visualizing a Decision Tree")

# Define smaller XGBoost model

st.markdown('''
XGBoost is an ensemble method and consists of multiple decision trees. In order to visualize one, a smaller decision tree is created with depth = 2.
The ellipses show the feature and the threshold used for splitting the data. These are connected to the next node using arrows. 
''')

dtc = DecisionTreeClassifier(max_depth = 2)
dtc.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(dtc, 
          max_depth = 2,
          filled=True, 
          feature_names=X_train.columns,
          impurity = False,
          class_names=['Not Interested', 'Interested'], 
          precision=2)
plt.show()
st.pyplot()


st.markdown('''
Finally, the single decision trees are combined using boosting.
In the figure, they are referred to as 'Classifier'.
''')

response = requests.get("https://raw.githubusercontent.com/majimaken/XAI-Classification/main/Boosting.png")
img = Image.open(BytesIO(response.content))

# Display the image using Streamlit
st.image(img, caption="Boosting")






