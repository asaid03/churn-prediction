import pandas as pd
import pickle
import os
from training.crossval import CrossValidator
from models.decision_tree import DecisionTree
from models.knn import NearestNeighbours
from models.elm import ELM

def load_data():
    X_train = pd.read_csv("dataset/X_train.csv").values
    y_train = pd.read_csv("dataset/y_train.csv").values.ravel()
    return X_train, y_train

def model_exists(model_name):
    return os.path.exists(f"saved_models/{model_name}.pkl")

def train_knn():
    X_train, y_train = load_data()
    k = 34
    knn = NearestNeighbours(neighbours=k)
    knn.fit(X_train, y_train)
    print(f"KNN Model with k={k} trained on full dataset")

    y_pred = knn.predict(X_train)

    from evaluator.model_evaluator import ModelEvaluator
    performance = ModelEvaluator.calculate_metrics(y_train, y_pred)
    
    # Save performance metrics
    performance_path = "saved_models/knn_k34_performance.pkl"
    with open(performance_path, "wb") as f:
        pickle.dump(performance, f)
    print(f"KNN performance saved to {performance_path}")


def train_decision_trees():
    X_train, y_train = load_data()
    uniformity_measures = ["gini", "entropy", "error"]

    os.makedirs("saved_models", exist_ok=True)

    for measure in uniformity_measures:
        model_name = f"decision_tree_{measure}"
        if model_exists(model_name):
            print(f"Skipping {model_name} - already trained.")
            continue

        dt_model = DecisionTree(uniformity_measure=measure, max_depth=10, min_samples_split=2)
        cv_results = CrossValidator.cross_validate(dt_model, X_train, y_train, folds=5, random_state=42)
        print(f"Cross-validation results for {measure}: {cv_results['mean_metrics']}")

        dt_model.fit(X_train, y_train)
        
        model_path = f"saved_models/{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(dt_model, f)
        print(f"{model_name} saved to {model_path}")
        

def train_elm():
    X_train,y_train = load_data()
    os.makedirs("saved_models", exist_ok=True)
    X_train, y_train = load_data()

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
            print(f"Skipping {model_name} - already trained.")
            continue

        elm = ELM(hidden_nodes=hidden_nodes, activation=activation, random_state=42)
        cv_results = CrossValidator.cross_validate(elm, X_train, y_train, folds=5, random_state=42)
        print(f"{model_name} CV Mean Metrics: {cv_results['mean_metrics']}")

        elm.fit(X_train, y_train)

        with open(f"saved_models/{model_name}.pkl", "wb") as f:
            pickle.dump(elm, f)

        with open(f"saved_models/{model_name}_cv.pkl", "wb") as f:
            pickle.dump(cv_results, f)

if __name__ == "__main__":
    train_decision_trees()
    train_knn()
    train_elm()