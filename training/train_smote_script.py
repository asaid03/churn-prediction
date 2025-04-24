"""
This script trains various models on the SMOTE-resampled dataset.
Utilies the CrossValidator class for cross-validation.
It includes Decision Trees, ELM, Logistic Regression, Weighted Logistic Regression, and Random Forest.
It saves the trained models and their cross-validation results to saved_models/smote.
"""

import pandas as pd
import pickle
import os

from training.crossval import CrossValidator

from models.random_forest import RandomForest
from models.decision_tree import DecisionTree
from models.elm import ELM
from models.logistic_regression import LogisticRegression
from models.weighted_lr import WeightedLogisticRegression



MODEL_DIR = "saved_models/smote"

def load_data():
    X_train = pd.read_csv("dataset/smote_resample/X_train.csv").values
    y_train = pd.read_csv("dataset/smote_resample/y_train.csv").values.ravel()
    return X_train, y_train

def model_exists(model_name):
    return os.path.exists(f"{MODEL_DIR}/{model_name}.pkl")


def train_decision_trees():
    X_train, y_train = load_data()
    uniformity_measures = ["gini", "entropy", "error"]
    
    os.makedirs(MODEL_DIR, exist_ok=True)

    for measure in uniformity_measures:
        model_name = f"dt_{measure}"
        if model_exists(model_name):
            print(f"Skipping {model_name}")
            continue

        dt_model = DecisionTree(uniformity_measure=measure, max_depth=5, min_samples_split=10)
        cv_results = CrossValidator.cross_validate(dt_model, X_train, y_train, folds=5, random_state=42)
        print(f"{model_name} CV: {cv_results['mean_metrics']}")

        dt_model.fit(X_train, y_train)

        with open(f"{MODEL_DIR}/{model_name}.pkl", "wb") as f:
            pickle.dump(dt_model, f)
            
        with open(f"{MODEL_DIR}/{model_name}_cv.pkl", "wb") as f:
            pickle.dump(cv_results, f)
            
        print(f"{model_name} saved")

def train_elm():
    X_train, y_train = load_data()
    os.makedirs(MODEL_DIR, exist_ok=True)

    configs = [
        {"activation": "relu", "hidden_nodes": 300},
        {"activation": "sigmoid", "hidden_nodes": 400},
        {"activation": "tanh", "hidden_nodes": 300},
    ]

    for config in configs:
        activation = config["activation"]
        hidden_nodes = config["hidden_nodes"]
        model_name = f"elm_{activation}"

        if model_exists(model_name):
            print(f"Skipping {model_name}")
            continue

        elm = ELM(hidden_nodes=hidden_nodes, activation=activation, random_state=42)
        cv_results = CrossValidator.cross_validate(elm, X_train, y_train, folds=5, random_state=42)
        print(f"{model_name} CV Mean Metrics: {cv_results['mean_metrics']}")

        elm.fit(X_train, y_train)

        with open(f"{MODEL_DIR}/{model_name}.pkl", "wb") as f:
            pickle.dump(elm, f)
        with open(f"{MODEL_DIR}/{model_name}_cv.pkl", "wb") as f:
            pickle.dump(cv_results, f)

def train_lr():
    X_train, y_train = load_data()
    os.makedirs(MODEL_DIR, exist_ok=True)

    lr = LogisticRegression(
        eta=0.001,
        epochs=5000,
        lambda_reg=0.0,
        threshold=0.495
    )
    
    cv_results = CrossValidator.cross_validate(lr, X_train, y_train, folds=5, random_state=42)
    print(f"lr CV: {cv_results['mean_metrics']}")

    lr.fit(X_train, y_train)

    with open(f"{MODEL_DIR}/lr.pkl", "wb") as f:
        pickle.dump(lr, f)
    with open(f"{MODEL_DIR}/lr_cv.pkl", "wb") as f:
        pickle.dump(cv_results, f)
    print("lr saved")

def train_wlr():
    X_train, y_train = load_data()
    os.makedirs(MODEL_DIR, exist_ok=True)

    cw = {0: 1, 1: 1.5}
    wlr = WeightedLogisticRegression(eta = 0.001,epochs=5000, lambda_reg = 0.0,threshold=0.495,class_weights= cw)
    cv_results = CrossValidator.cross_validate(wlr, X_train, y_train, folds=5, random_state=42)
    print(f"wlr CV: {cv_results['mean_metrics']}")

    wlr.fit(X_train, y_train)

    with open(f"{MODEL_DIR}/wlr.pkl", "wb") as f:
        pickle.dump(wlr, f)
    with open(f"{MODEL_DIR}/wlr_cv.pkl", "wb") as f:
        pickle.dump(cv_results, f)
    print("wlr saved")

def train_rf():
    X_train, y_train = load_data()
    os.makedirs(MODEL_DIR, exist_ok=True)

    rf = RandomForest(n_estimators=100, max_depth=5, min_samples_split=10)
    cv_results = CrossValidator.cross_validate(rf, X_train, y_train, folds=5, random_state=42)
    print(f"rf CV: {cv_results['mean_metrics']}")

    rf.fit(X_train, y_train)

    with open(f"{MODEL_DIR}/rf.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(f"{MODEL_DIR}/rf_cv.pkl", "wb") as f:
        pickle.dump(cv_results, f)
    print("rf saved")

if __name__ == "__main__":
    train_lr()
    train_wlr()
    #train_elm()
    #train_rf()
    #train_decision_trees()