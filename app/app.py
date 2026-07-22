from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import streamlit as st
import numpy as py
import matplotlib.pyplot as mlt
import seaborn as sns
# **Telecom Customer Churn Prediction**
# import library
# read dataset
df = pd.read_csv(r'C:\Users\ma164\OneDrive\Desktop\NA\cleaned_dataset.csv')
df.head()
df.info()
df.describe()
df.isna().sum()
df.duplicated().sum()
df.columns
df.drop(columns=['customerid'], inplace=True)
df.churn.value_counts()
### ***Encoding***

#### ***Label encoding***

#### ***Split the data to features and target***

# define y and X
y = df['churn']
X = df.drop(['churn'],axis=1)
#### ***Split the data to train and test***
# train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#### ***Encoding***
# Label encoding
from sklearn.preprocessing import LabelEncoder

label_encoders = {}

for col in X_train.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

### ***Standard scaling***

# standard scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y.value_counts()
#### ***Handle Unbalance data***

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42, k_neighbors=5) 
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
y_train.value_counts()
y_train_smote.value_counts()
y.value_counts()
### ***Modeling***

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42)
# train the model
model.fit(X_train_smote, y_train_smote)
# test the model
y_pred = model.predict(X_test)
# R2 Score between y_test and y_pred
from sklearn.metrics import accuracy_score,mean_squared_error
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# use joblib to save the model
import joblib
joblib.dump(model, "logistic_regression_model.pkl")
print("Model saved as logistic_regression_model.pkl")

st.write("joblib loaded successfully") 
@ st.cache_data
def load_data():
    df= pd.read_csv("cleaned_dataset.csv")
    if 'customerid' in df.columns:
        df.drop(columns=['customerid'], inplace=True)
    df_dashboard = df.copy()
    df_ml = df.copy()
    df_ml['churn'] = df_ml['churn'].map({'Yes': 1, 'No': 0})
    categorical_cols = [
    'gender', 'partner', 'dependents', 'phoneservice', 'multiplelines',
    'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
    'streamingtv', 'streamingmovies', 'paperlessbilling',
    'internetservice', 'contract', 'paymentmethod'
    ]
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col])
        encoders[col] = le
    scaler = StandardScaler()
    num_cols = ["tenure", "monthlycharges", "totalcharges"]
    df_ml[num_cols] = scaler.fit_transform(df_ml[num_cols])
    X = df_ml.drop('churn', axis=1)
    y = df_ml['churn']
    return df_dashboard, X, y, encoders,scale
@st.cache_resource
def load_model():
    return joblib.load("logistic_regression_model.pkl")
def analysis_page(df):
    st.title(" Telecom Churn Analysis Dashboard")
    col1, col2 = st.columns(2)
    col1.metric("Total Customers", len(df))
    col2.metric("Churn Rate", f"{(df['churn'] == 'Yes').mean()*100:.2f}%")
    st.divider()
    st.subheader("Payment Method Distribution")
    payment_counts = df['paymentmethod'].value_counts().reset_index()
    payment_counts.columns = ['Payment Method', 'Count']
    st.plotly_chart(
        px.bar(payment_counts, x='Payment Method', y='Count'),
        use_container_width=True
)
    st.subheader("Contract Types")
    st.plotly_chart(
        px.pie(df, names='contract'),
        use_container_width=True
)
st.subheader("Churn Distribution")
st.plotly_chart(
px.pie(df, names='churn'),
use_container_width=True
)
st.subheader("Contract vs Churn")
st.plotly_chart(
px.histogram(df, x='contract', color='churn', barmode='group')