import os
import pickle
import pandas as pd
from evaluator.model_evaluator import ModelEvaluator

class AppModel:
    def __init__(self):
        self.models = {}
        self.performance_data = {}
        self.observers = []
        
        # hard loading models
        model_files = {
            'Decision Tree (Entropy)': r"C:\PROJECT\saved_models\decision_tree_entropy.pkl",
            'Decision Tree (Error)': r"C:\PROJECT\saved_models\decision_tree_error.pkl",
            'Decision Tree (Gini)': r"C:\PROJECT\saved_models\decision_tree_gini.pkl",
        }

        # Load test dataset
        self.X_test, self.y_test = self.load_test_dataset()

        for name, path in model_files.items():
            try:
                with open(path, "rb") as f:
                    model = pickle.load(f)
                self.models[name] = model

                # check test dataset is loaded
                if self.X_test is not None and self.y_test is not None:
                    y_pred = model.predict(self.X_test)  # Run model on test set
                    self.performance_data[name] = ModelEvaluator.calculate_metrics(self.y_test, y_pred)
                else:
                    self.performance_data[name] = {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0 , "f2_score": 0}

            except Exception as e:
                print(f"Error loading model '{name}' from {path}: {e}")

        

    def load_test_dataset(self):
        dataset_path = r"C:\PROJECT\dataset"
        try:
            # Load X_test and y_test
            X_test = pd.read_csv(os.path.join(dataset_path, "X_test.csv")).values  # Convert to NumPy array
            y_test = pd.read_csv(os.path.join(dataset_path, "y_test.csv")).values.ravel()  # Flatten to 1D array

            return X_test, y_test
        except Exception as e:
            print(f"Error loading test dataset: {e}")
            return None, None

    def add_observer(self, observer):
        if observer not in self.observers:
            self.observers.append(observer)

    def remove_observer(self, observer):
        if observer in self.observers:
            self.observers.remove(observer)

    def notify_observers(self, data):
        for observer in self.observers:
            observer.update(data)

    def get_performance(self, selected_models):
        performance = {}
        for model in selected_models:
            performance[model] = self.performance_data.get(model, {})
        return performance
