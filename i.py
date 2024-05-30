# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Suppressing warnings
import warnings
warnings.filterwarnings("ignore") 

# Loading data
train = pd.read_csv(r'C:\Users\Preti Sherine\Downloads\salary-prediction-for-job-postings\usjobs_train.csv')
test = pd.read_csv(r'C:\Users\Preti Sherine\Downloads\salary-prediction-for-job-postings\usjobs_test.csv')

# Save ID columns
train_ID = train['ID']
test_ID = test['ID']

# Drop ID columns from original datasets
train.drop('ID', axis=1, inplace=True)
test.drop('ID', axis=1, inplace=True)

# Extracting the target variable
y_train = train['Mean_Salary'].values

# Drop the target variable from the training data
train.drop('Mean_Salary', axis=1, inplace=True)

# Define MAPE function
def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    y_true, y_pred = y_true[mask], y_pred[mask]
    percentage_error = np.abs((y_true - y_pred) / y_true)
    mape = np.mean(percentage_error) * 100
    return mape

# Drop rows with missing values from both train and test datasets
train.dropna(inplace=True)
y_train = y_train[train.index]  # Update y_train accordingly
test.dropna(inplace=True)

# Convert categorical variables to one-hot encoding
train = pd.get_dummies(train)

# Align the columns of the test set with those of the training set
train, test = train.align(pd.get_dummies(test), join='inner', axis=1)

# Initialize and train Linear Regression model
model = LinearRegression()
model.fit(train, y_train)

# Predict on test set
test_pred = model.predict(test)

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Salary Prediction for Job Posting")

# Column layout for the main content and additional information on the right side
col1, col2 = st.columns([4, 1])

# Main content: Display the top 100 predictions in a table
with col1:
    st.markdown("---")
    st.title("Predicted Salaries")
    # Create a DataFrame with ID, Job, and Predicted Salary
    predictions_df = pd.DataFrame({
        "ID": test_ID[:100], 
        "Job": test.columns[3:][0:100],  # Assuming the first 3 columns are non-job related features
        "Predicted Salary": test_pred[:100].astype(int)  # Convert to int to remove decimals
    })  
    # Increase the size of the output table
    st.write(predictions_df.style.set_table_styles([{
        'selector': 'table',
        'props': [('width', '100%'), ('font-size', '40px')]
    }]))

# Additional information on the right side
with col2:
    st.title("PRETI SHERINE P")
    st.title("BATCH D100")
    st.title("DATA SET FROM KAGGLE")
