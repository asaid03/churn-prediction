import os
import pickle
import pandas as pd
from evaluator.model_evaluator import ModelEvaluator
from gui.config import MODEL_FILES, X_TEST_PATH, Y_TEST_PATH ,CV_FILES

class AppModel:
    def __init__(self):
        self.models = {}
        self.performance_data = {}
        self.cv_scores = {}
        self.observers = []

        # Load test dataset
        self.X_test, self.y_test = self.load_test_dataset()
        self._load_models()

    def load_test_dataset(self):
        """Load test dataset from configured paths."""
        try:
            X_test = pd.read_csv(X_TEST_PATH).values
            y_test = pd.read_csv(Y_TEST_PATH).values.ravel()
            return X_test, y_test
        except Exception as e:
            print(f"Error loading test dataset: {e}")
            return None, None

    def _load_models(self):
        """Load pre-trained models and KNN performance from configured file paths."""
        for name, path in MODEL_FILES.items():
            try:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        model = pickle.load(f)

                    if name == "KNN (k=34)":
                        self.models[name] = None
                        self.performance_data[name] = model
                        print(f"{name} performance loaded successfully")
                    else:
                        self.models[name] = model
                        self.performance_data[name] = self._evaluate_model(model)
                        print(f"{name} model loaded successfully")
                        
                    if name in CV_FILES:
                        with open(CV_FILES[name], "rb") as f:
                            self.cv_scores[name] = pickle.load(f)
                            print(f"{name} CV scores loaded successfully")
                
                else:
                    print(f"Warning: Model file not found at {path}")
            except (pickle.UnpicklingError, FileNotFoundError, Exception) as e:
                print(f"Error loading model '{name}' from {path}: {e}")


    def _evaluate_model(self, model):
        """Evaluate model on test dataset if available."""
        if self.X_test is not None and self.y_test is not None:
            y_pred = model.predict(self.X_test)
            return ModelEvaluator.calculate_metrics(self.y_test, y_pred)
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0, "f2_score": 0}

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
        """Retrieve performance metrics for selected models."""
        performance = {}
        for model in selected_models:
            if model in self.performance_data:
                performance[model] = self.performance_data[model]
            else:
                performance[model] = {}
        return performance

    def get_cv_scores(self, selected_models):
        cv_scores = {}
        for model in selected_models:
            cv_scores[model] = self.cv_scores.get(model, {})
        return cv_scores

