import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

@staticmethod
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

import numpy as np

class LogisticRegression:
    def __init__(self, eta=0.001, epochs=1000, lambda_reg=0.01):
        """
        Initialise Logistic Regression model.

        Parameters:
        - eta: Learning rate (default: 0.001)
        - epochs: Number of training epochs (default: 1000)
        - lambda_reg: L2 regularisation parameter (default: 0.01)
        """
        self.eta = eta
        self.epochs = epochs
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None

    def sigmoid(self, x):
        """
        Sigmoid activation function.It squashes the logits to a range between 0 and 1.

        Parameters:
            self: Instance of the class
            x: Logits
    
        Returns:
            Sigmoid of x
        """
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        """
        Train the Logistic Regression model.

        Parameters:
            X: Training data features
            y: Target data labels 
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialise weights to zeros
        self.bias = 0  # Initialise bias to zero

        for _ in range(self.epochs):
            logits = np.dot(X, self.weights) + self.bias
            probabilites = self.sigmoid(logits)

            # we compute gradients with L2 regularisation
            dw = (1 / n_samples) * (np.dot(X.T, (probabilites - y)) + self.lambda_reg * self.weights) 
            db = (1 / n_samples) * np.sum(probabilites - y)

            # Update weights and bias
            self.weights = self.weights - self.eta * dw
            self.bias = self.bias - self.eta * db

    def predict(self, X):
        """
        Predict class labels for input data.

        Parameters:
            X: Training data features

        Returns:
            Predicted class labels (0 or 1)
        """
        logits = np.dot(X, self.weights) + self.bias 
        y_pred = self.sigmoid(logits)
        class_predictions = [0 if y <= 0.5 else 1 for y in y_pred]  # Threshold at 0.5
        return class_predictions

    def compute_loss(self, X, y):
        """
        Compute the logistic loss with L2 regularisation.

        Parameters:
            X: Input data features
            y: Target data labels

        Returns:
        - Loss value
        """
        logits = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(logits)
        # Logistic loss
        loss = -np.mean(y * np.log(probabilities + 1e-10) + (1 - y) * np.log(1 - probabilities + 1e-10)) # Add epsilon to prevent log(0)
        # Add L2 regularisation term
        l2_penalty = (self.lambda_reg / 2) * np.sum(self.weights**2)
        return loss + l2_penalty # Add L2 regularisation term

    
    
