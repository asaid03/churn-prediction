import pandas as pd
import numpy as np

def load_data(file_path):
    churn_data = pd.read_csv(file_path)
    churn_data = churn_data.drop(['customerID'], axis=1)
    
    # Replace empty strings with NaN and convert to float
    churn_data['TotalCharges'] = churn_data['TotalCharges'].replace(" ", np.nan)
    churn_data['TotalCharges'] = churn_data['TotalCharges'].astype(float)
    
    return churn_data

def compute_total_charges(row):
    if pd.isnull(row['TotalCharges']) and row['MonthlyCharges'] > 0:
        return row['tenure'] * row['MonthlyCharges']
    return row['TotalCharges']

def preprocess_data(churn_data):
    churn_data['TotalCharges'] = churn_data.apply(compute_total_charges, axis=1)
    
    # Identify categorical columns
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    non_binary_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaymentMethod']
    
    binary_mappings = {}
    one_hot_categories = {}
    
    # Encode binary categorical variables
    for col in binary_cols:
        unique_values = churn_data[col].unique()
        mapping = {unique_values[0]: 0, unique_values[1]: 1}
        binary_mappings[col] = mapping
        churn_data[col] = churn_data[col].map(mapping)
    
    # One-hot encode non-binary categorical variables
    for col in non_binary_cols:
        one_hot_categories[col] = churn_data[col].unique().tolist()
        for value in churn_data[col].unique():
            churn_data[f"{col}_{value}"] = (churn_data[col] == value).astype(int)
        churn_data.drop(col, axis=1, inplace=True)
    
    return churn_data

def train_test_split(X, y, test_size=0.2, random_seed=None):
    np.random.seed(random_seed)  # For reproducibility
    indices = np.random.permutation(len(X))  # Shuffle the indices
    test_set_size = int(len(X) * test_size)
    
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    
    return X_train, X_test, y_train, y_test

def min_max_scaling(X_train, X_test):
    min_value = X_train.min(axis=0)
    max_value = X_train.max(axis=0)
    
    X_train = (X_train - min_value) / (max_value - min_value)
    X_test = (X_test - min_value) / (max_value - min_value)
    
    return X_train, X_test

if __name__ == "__main__":
    file_path = 'dataset/Telco-Customer-Churn.csv'
    churn_data = load_data(file_path)
    churn_data = preprocess_data(churn_data)
    
    X = churn_data.drop(columns=['Churn'])
    y = churn_data['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=42)
    X_train, X_test = min_max_scaling(X_train, X_test)
    
    # Save preprocessed data
    X_train.to_csv('dataset/X_train.csv', index=False)
    X_test.to_csv('dataset/X_test.csv', index=False)
    y_train.to_csv('dataset/y_train.csv', index=False)
    y_test.to_csv('dataset/y_test.csv', index=False)
