import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the dataset
@st.cache_resource
def load_data():
    data = pd.read_csv("creditcard.csv")
    return data

data = load_data()

# Streamlit UI
st.title("Credit Fraud Detection")

st.sidebar.header("Data Exploration")
st.sidebar.subheader("Data Overview")
st.write("### Data Preview")
st.write(data.head())

st.sidebar.subheader("Data Statistics")
st.write("### Data Shape")
st.write(data.shape)
st.write("### Data Description")
st.write(data.describe())

# Display Fraud and Valid Transactions
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

st.write("### Fraud Cases: {}".format(len(fraud)))
st.write("### Valid Transactions: {}".format(len(valid)))
st.write("### Outlier Fraction: {:.4f}".format(len(fraud) / float(len(valid))))

st.write("### Amount Details of Fraudulent Transactions")
st.write(fraud.Amount.describe())
st.write("### Amount Details of Valid Transactions")
st.write(valid.Amount.describe())

# Plot Correlation Matrix
st.write("### Correlation Matrix")
corrmat = data.corr()
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, ax=ax)
st.pyplot(fig)

# Model Training and Evaluation
st.sidebar.header("Model Training")
st.sidebar.subheader("Train Model")
if st.sidebar.button("Train Model"):
    X = data.drop(['Class'], axis=1)
    Y = data["Class"]

    xData = X.values
    yData = Y.values

    xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=42)

    rfc = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1)
    rfc.fit(xTrain, yTrain)

    yPred = rfc.predict(xTest)

    acc = accuracy_score(yTest, yPred)
    prec = precision_score(yTest, yPred)
    rec = recall_score(yTest, yPred)
    f1 = f1_score(yTest, yPred)
    MCC = matthews_corrcoef(yTest, yPred)

    st.write("### Model Evaluation")
    st.write("Accuracy: {:.4f}".format(acc))
    st.write("Precision: {:.4f}".format(prec))
    st.write("Recall: {:.4f}".format(rec))
    st.write("F1-Score: {:.4f}".format(f1))
    st.write("Matthews Correlation Coefficient: {:.4f}".format(MCC))

    # Plot Confusion Matrix
    LABELS = ['Normal', 'Fraud']
    conf_matrix = confusion_matrix(yTest, yPred)
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)
