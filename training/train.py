import pandas as pd
import pickle
import os
from models.decision_tree import DecisionTree  

# Load preprocessed data
def load_data():
    X_train = pd.read_csv('dataset/X_train.csv').values
    y_train = pd.read_csv('dataset/y_train.csv').values.ravel()
    return X_train, y_train

# Trains and saves Decision Tree models with different uniformity measures
def train_decision_trees():
    X_train, y_train = load_data()
    uniformity_measures = ["gini", "entropy", "error"]
    
    os.makedirs("saved_models", exist_ok=True)
    
    for measure in uniformity_measures:
        dt_model = DecisionTree(uniformity_measure=measure, max_depth=10, min_samples_split=2)
        dt_model.fit(X_train, y_train)
        
        path = f"saved_models/decision_tree_{measure}.pkl"
        with open(path, "wb") as f:
            pickle.dump(dt_model, f)
        print(f"Decision Tree model saved to {path}")

if __name__ == "__main__":
    train_decision_trees()