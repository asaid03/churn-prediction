'''
This contains the configuration for the GUI application.
It includes the paths to the dataset, model files, and other resources needed for the application.
'''

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
X_TEST_PATH = os.path.join(DATASET_DIR, "X_test.csv")
Y_TEST_PATH = os.path.join(DATASET_DIR, "y_test.csv")

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
SMOTE_DIR = os.path.join(MODEL_DIR, "smote")
TOMEK_DIR = os.path.join(MODEL_DIR, "tomek")

import os

MODEL_DIR = "saved_models"

MODEL_FILES = {
    'Decision Tree (Entropy)': os.path.join(MODEL_DIR, "dt_entropy.pkl"),
    'Decision Tree (Error)': os.path.join(MODEL_DIR, "dt_error.pkl"),
    'Decision Tree (Gini)': os.path.join(MODEL_DIR, "dt_gini.pkl"),
    
    'Logistic Regression': os.path.join(MODEL_DIR, "lr.pkl"),
    'Weighted Logistic Regression': os.path.join(MODEL_DIR, "wlr.pkl"),
    
    'Random Forest': os.path.join(MODEL_DIR, "rf.pkl"),
    "KNN (k=34)": os.path.join(MODEL_DIR, "knn_k34_performance.pkl"),
    
    'ELM (ReLU)': os.path.join(MODEL_DIR, "elm_relu.pkl"),
    'ELM (Sigmoid)': os.path.join(MODEL_DIR, "elm_sigmoid.pkl"),
    'ELM(Tanh)': os.path.join(MODEL_DIR, "elm_tanh.pkl"),
    
    # SMOTE models
    'Decision Tree (Entropy) [SMOTE]': os.path.join(SMOTE_DIR, "dt_entropy.pkl"),
    'Decision Tree (Error) [SMOTE]': os.path.join(SMOTE_DIR, "dt_error.pkl"),
    'Decision Tree (Gini) [SMOTE]': os.path.join(SMOTE_DIR, "dt_gini.pkl"),
    
    'Weighted Logistic Regression [SMOTE]': os.path.join(SMOTE_DIR, "wlr.pkl"),
    'Logistic Regression [SMOTE]': os.path.join(SMOTE_DIR, "lr.pkl"),
    
    'Random Forest [SMOTE]': os.path.join(SMOTE_DIR, "rf.pkl"),
    
    'ELM (ReLU) [SMOTE]': os.path.join(SMOTE_DIR, "elm_relu.pkl"),
    'ELM (Sigmoid) [SMOTE]': os.path.join(SMOTE_DIR, "elm_sigmoid.pkl"),
    'ELM(Tanh) [SMOTE]': os.path.join(SMOTE_DIR, "elm_tanh.pkl"),

    # Tomek models
    'Decision Tree (Entropy) [Tomek]': os.path.join(TOMEK_DIR, "dt_entropy.pkl"),
    'Decision Tree (Error) [Tomek]': os.path.join(TOMEK_DIR, "dt_error.pkl"),
    'Decision Tree (Gini) [Tomek]': os.path.join(TOMEK_DIR, "dt_gini.pkl"),
    
    'Weighted Logistic Regression [Tomek]': os.path.join(TOMEK_DIR, "wlr.pkl"),
    'Logistic Regression [Tomek]': os.path.join(TOMEK_DIR, "lr.pkl"),
    
    'Random Forest [Tomek]': os.path.join(TOMEK_DIR, "rf.pkl"),
    'ELM (ReLU) [Tomek]': os.path.join(TOMEK_DIR, "elm_relu.pkl"),
    'ELM (Sigmoid) [Tomek]': os.path.join(TOMEK_DIR, "elm_sigmoid.pkl"),
    'ELM(Tanh) [Tomek]': os.path.join(TOMEK_DIR, "elm_tanh.pkl")
    
}

CV_FILES = {
    'Decision Tree (Entropy)': os.path.join(MODEL_DIR, "dt_entropy_cv.pkl"),
    'Decision Tree (Error)': os.path.join(MODEL_DIR, "dt_error_cv.pkl"),
    'Decision Tree (Gini)': os.path.join(MODEL_DIR, "dt_gini_cv.pkl"),

    'Logistic Regression': os.path.join(MODEL_DIR, "lr_cv.pkl"),
    'Weighted Logistic Regression': os.path.join(MODEL_DIR, "wlr_cv.pkl"),
    
    'Random Forest': os.path.join(MODEL_DIR, "rf_cv.pkl"),

    'ELM (ReLU)': os.path.join(MODEL_DIR, "elm_relu_cv.pkl"),
    'ELM (Sigmoid)': os.path.join(MODEL_DIR, "elm_sigmoid_cv.pkl"),
    'ELM(Tanh)': os.path.join(MODEL_DIR, "elm_tanh_cv.pkl"),

    # SMOTE 
    'Decision Tree (Entropy) [SMOTE]': os.path.join(SMOTE_DIR, "dt_entropy_cv.pkl"),
    'Decision Tree (Error) [SMOTE]': os.path.join(SMOTE_DIR, "dt_error_cv.pkl"),
    'Decision Tree (Gini) [SMOTE]': os.path.join(SMOTE_DIR, "dt_gini_cv.pkl"),

    'Logistic Regression [SMOTE]': os.path.join(SMOTE_DIR, "lr_cv.pkl"),
    'Weighted Logistic Regression [SMOTE]': os.path.join(SMOTE_DIR, "wlr_cv.pkl"),
    
    'Random Forest [SMOTE]': os.path.join(SMOTE_DIR, "rf_cv.pkl"),

    'ELM (ReLU) [SMOTE]': os.path.join(SMOTE_DIR, "elm_relu_cv.pkl"),
    'ELM (Sigmoid) [SMOTE]': os.path.join(SMOTE_DIR, "elm_sigmoid_cv.pkl"),
    'ELM(Tanh) [SMOTE]': os.path.join(SMOTE_DIR, "elm_tanh_cv.pkl"),

    # Tomek 
    'Decision Tree (Entropy) [Tomek]': os.path.join(TOMEK_DIR, "dt_entropy_cv.pkl"),
    'Decision Tree (Error) [Tomek]': os.path.join(TOMEK_DIR, "dt_error_cv.pkl"),
    'Decision Tree (Gini) [Tomek]': os.path.join(TOMEK_DIR, "dt_gini_cv.pkl"),

    'Logistic Regression [Tomek]': os.path.join(TOMEK_DIR, "lr_cv.pkl"),
    'Weighted Logistic Regression [Tomek]': os.path.join(TOMEK_DIR, "wlr_cv.pkl"),
    
    'Random Forest [Tomek]': os.path.join(TOMEK_DIR, "rf_cv.pkl"),

    'ELM (ReLU) [Tomek]': os.path.join(TOMEK_DIR, "elm_relu_cv.pkl"),
    'ELM (Sigmoid) [Tomek]': os.path.join(TOMEK_DIR, "elm_sigmoid_cv.pkl"),
    'ELM(Tanh) [Tomek]': os.path.join(TOMEK_DIR, "elm_tanh_cv.pkl")
}

