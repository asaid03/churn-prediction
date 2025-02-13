import pickle
import pandas as pd
import os
from evaluator.model_evaluator import ModelEvaluator

def load_test_data():
    X_test = pd.read_csv('dataset/X_test.csv').values
    y_test = pd.read_csv('dataset/y_test.csv').values.ravel()
    return X_test, y_test

def main():
    # Load pre-trained models
    model_paths = {
        "Decision_Tree_Gini": "saved_models/decision_tree_gini.pkl",
        "Decision_Tree_Entropy": "saved_models/decision_tree_entropy.pkl",
        "Decision_Tree_Error": "saved_models/decision_tree_error.pkl"
    }

    # Load test data
    X_test, y_test = load_test_data()

    model_perfromance = {}

    # Ensure checkpoints directory exists
    os.makedirs("checkpoints", exist_ok=True)

    # Evaluate each model
    for model_name, model_path in model_paths.items():
        with open(model_path, "rb") as file:
            model = pickle.load(file)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics using ModelEvaluator
        metrics = ModelEvaluator.calculate_metrics(y_test, y_pred)
        model_perfromance[model_name] = metrics

        # Save perofmance metrics to a file
        metrics_path = os.path.join("checkpoints", f"{model_name}_metrics.pkl")
        with open(metrics_path, "wb") as f:
            pickle.dump(metrics, f)
        print(f"Saved performance results for {model_name} to {metrics_path}")

if __name__ == "__main__":
    main()


