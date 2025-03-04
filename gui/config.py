import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
X_TEST_PATH = os.path.join(DATASET_DIR, "X_test.csv")
Y_TEST_PATH = os.path.join(DATASET_DIR, "y_test.csv")

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
MODEL_FILES = {
    'Decision Tree (Entropy)': os.path.join(MODEL_DIR, "decision_tree_entropy.pkl"),
    'Decision Tree (Error)': os.path.join(MODEL_DIR, "decision_tree_error.pkl"),
    'Decision Tree (Gini)': os.path.join(MODEL_DIR, "decision_tree_gini.pkl"),
    'KNN (k=34)': os.path.join(MODEL_DIR, "knn_k34_performance.pkl")  
}
