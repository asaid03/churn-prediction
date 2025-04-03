import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
X_TEST_PATH = os.path.join(DATASET_DIR, "X_test.csv")
Y_TEST_PATH = os.path.join(DATASET_DIR, "y_test.csv")

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

import os

MODEL_DIR = "saved_models"

MODEL_FILES = {
    'Decision Tree (Entropy)': os.path.join(MODEL_DIR, "dt_entropy.pkl"),
    'Decision Tree (Error)': os.path.join(MODEL_DIR, "dt_error.pkl"),
    'Decision Tree (Gini)': os.path.join(MODEL_DIR, "dt_gini.pkl"),
    
    'Logistic Regression': os.path.join(MODEL_DIR, "lr.pkl"),
    'Weighted Logistic Regression': os.path.join(MODEL_DIR, "wlr.pkl"),
    'Random Forest': os.path.join(MODEL_DIR, "rf.pkl"),
    
    'K-Nearest Neighbours (k=34)': os.path.join(MODEL_DIR, "knn_k34_performance.pkl"),

    'ELM (ReLU)': os.path.join(MODEL_DIR, "elm_relu.pkl"),
    'ELM (Sigmoid)': os.path.join(MODEL_DIR, "elm_sigmoid.pkl"),
    'ELM(Tanh)': os.path.join(MODEL_DIR, "elm_tanh.pkl")
}
