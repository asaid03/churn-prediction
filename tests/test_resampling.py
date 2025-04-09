import numpy as np
import pandas as pd
from models.logistic_regression import LogisticRegression
from evaluator.model_evaluator import ModelEvaluator
from preprocessing.smote import Smote
from preprocessing.tomek import TomekLinks


"""
    Regression tests for the resampling methods (SMOTE and Tomek Links)
    Checks if SMOTE balances the classes, 
    Chekcs if Tomek reduces the majority class samples,
    Checks if the model performance improves after applying SMOTE.
"""
# Load data
X_train_full = pd.read_csv("dataset/X_train.csv").values
y_train_full = pd.read_csv("dataset/y_train.csv").values.ravel()
X_test = pd.read_csv("dataset/X_test.csv").values
y_test = pd.read_csv("dataset/y_test.csv").values.ravel()

# reduce the sample size
sample_size = 800
r = np.random.default_rng(seed=42)
i = r.choice(len(X_train_full), size=sample_size, replace=False)
X_train = X_train_full[i]
y_train = y_train_full[i]

def test_smote_balances_classes():
    smote = Smote(k_neighbours=5, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    counts = np.bincount(y_res)
    assert counts[0] == counts[1], f"class imbalance after applying SMOTE: {counts}"

def test_tomek_reduces_majority_class(): 
    tomek = TomekLinks()
    X_res, y_res = tomek.fit_resample(X_train, y_train)
    original_majority = np.bincount(y_train)[0]
    new_majority = np.bincount(y_res)[0]
    assert new_majority < original_majority, f"Tomek did not reduce majority class samples"

def test_smote_model_performance():
    smote = Smote(k_neighbours=5, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    model = LogisticRegression()
    model.fit(X_res, y_res)
    y_pred = model.predict(X_test)

    metrics = ModelEvaluator.calculate_metrics(y_test, y_pred)
    assert metrics["recall"] >= 0.6, f"Recall too low after SMOTE: {metrics['recall']}"


if __name__ == "__main__":
    test_smote_balances_classes()
    test_tomek_reduces_majority_class()
    test_smote_model_performance()
    print("SMOTE & Tomek regression tests passed.")
