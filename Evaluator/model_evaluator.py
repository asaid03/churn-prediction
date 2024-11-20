
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ModelEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def accuracy_score(y_true, y_pred):
        """
        Calculates the accuracy score of predictions.

        Parameters:
        - y_true (np-array): The true labels.
        - y_pred (np-array): The predicted labels.

        Returns:
        - float: The accuracy score of correct predictions.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if y_true.shape != y_pred.shape:
            raise ValueError("Mismatch of the number of predicitions and true labels")
        
        return np.mean(y_true == y_pred)
    
    

    @staticmethod
    def precision_score(y_true, y_pred):
        """
        Calculate the precision score of predictions.

        Parameters:
        - y_true (np-array): The true labels.
        - y_pred (np-array): The predicted labels.

        Returns:
        - float: The precision score or 0 if there are no predicted positives.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if y_true.shape != y_pred.shape:
            raise ValueError("Mismatch of the number of predicitions and true labels")
        
        # True Positives: predcitions that are positive and true
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        
        # Predicted Positives: Total number of predictions that are 1
        predicted_positives = np.sum(y_pred == 1)
        
        if predicted_positives != 0:
            return true_positives / predicted_positives

        return 0



    @staticmethod
    def recall_score(y_true, y_pred):
        """
        Calculate the recall score of predictions.

        Parameters:
        - y_true (np-array): The true labels.
        - y_pred (np-array): The predicted labels.

        Returns:
        - float: The recall score, or 0 if there are no actual positives.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if y_true.shape != y_pred.shape:
            raise ValueError("Mismatch of the number of predictions and true labels")
        
        # True Positives: Predictions that are positive and correct
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        
        # Actual Positives: Total number of actual true positives
        actual_positives = np.sum(y_true == 1)
        
        # Avoid division by zero
        if actual_positives != 0:
            return true_positives / actual_positives

        return 0



    @staticmethod
    def f1_score(y_true, y_pred):
        """
        Calculate the F1-score of predictions.

        Parameters:
        - y_true (np-array): The true labels.
        - y_pred (np-array): The predicted labels.

        Returns:
        - float: Returns the F1-score  or 0 if both precision and recall are 0.
        """
        # Calculate precision and recall using my existing methods
        precision = ModelEvaluator.precision_score(y_true, y_pred)
        recall = ModelEvaluator.recall_score(y_true, y_pred)
        
        # Avoid division by zero
        if precision == 0 and recall == 0:
            return 0
        
        # F1-score formula
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1


    @staticmethod
    def f2_score(y_true, y_pred):
        """
        Calculate the F2-score of predictions, focuses recall over precision as false negatives are more important in churn prediction.

        Parameters:
        - y_true (np-array): The true labels.
        - y_pred (np-array): The predicted labels.

        Returns:
        - float: Returns the F2-score or 0 if both precision and recall are 0.
        """
        # Calculate precision and recall using my existing methods
        precision = ModelEvaluator.precision_score(y_true, y_pred)
        recall = ModelEvaluator.recall_score(y_true, y_pred)
        
        # Avoid division by zero
        if precision == 0 and recall == 0:
            return 0
        
        # F2-score formula
        f2 = (1 + 2**2) * (precision * recall) / ((2**2 * precision) + recall)
        return f2



    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        Calculates evaluation metrics using ModelEvaluator methods.

        Parameters:
        - y_true: The true labels.
        - y_pred: The Predicted labels.

        Returns:
        - dict: A dictionary of evaluation metrics.
        """
        return {
            "accuracy": ModelEvaluator.accuracy_score(y_true, y_pred),
            "precision": ModelEvaluator.precision_score(y_true, y_pred),
            "recall": ModelEvaluator.recall_score(y_true, y_pred),
            "f1_score": ModelEvaluator.f1_score(y_true, y_pred),
            "f2_score": ModelEvaluator.f2_score(y_true, y_pred),
        }
        
    @staticmethod
    def compare_and_visualise_metrics(file_paths, labels):
        """
        Compares and visualises evaluation metrics stored in the checkpoint files using Pandas.

        Parameters:
            file_paths (list of str): Paths to the pickle files containing metrics.
            labels (list of str): Labels for the scaling techniques/models being compared.
        """
        if len(file_paths) != len(labels):
            raise ValueError("The number of file paths and labels must match.")

        # Load metrics 
        metrics_data = {}
        for file_path, label in zip(file_paths, labels):
            with open(file_path, 'rb') as f:
                metrics_data[label] = pickle.load(f)

        # Convert metrics_data to a DataFrame
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.index.name = "Metric"

        #plotting
        metrics_df.plot(kind="bar", figsize=(10, 6))
        plt.title("Comparison of Evaluation Metrics")
        plt.ylabel("Scores")
        plt.xlabel("Metrics")
        plt.xticks(rotation=0)
        plt.legend(title="Scaling Techniques")
        plt.tight_layout()
        plt.show()

