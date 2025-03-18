import numpy as np
from models.logistic_regression import LogisticRegression

class WeightedLogisticRegression(LogisticRegression):
    def __init__(self, eta=0.001, epochs=1000, lambda_reg=0.01, threshold=0.5, class_weights=None):
        """
        Initialises a Weighted Logistic Regression model.

        Parameters:
        - eta: Learning rate (default: 0.001)
        - epochs: Number of training epochs (default: 1000)
        - lambda_reg: L2 regularisation parameter (default: 0.01)
        - threshold: Decision threshold for binary classification (default: 0.5)
        - class_weights: Dictionary of class weights (e.g., {0: 1, 1: 5})
        """
        super().__init__(eta, epochs, lambda_reg, threshold)  
        self.class_weights = class_weights 

    def fit(self, X, y):
        """
        Train the Weighted Logistic Regression model.

        Parameters:
            X: Training data features
            y: Target data labels 
        """
        # Initialise weights and bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  
        self.bias = 0  

        # If class_weights is None, assign equal weights to both classes
        if self.class_weights is None:
            self.class_weights = {0: 1, 1: 1}

        # Initialise sample weights array
        sample_weights = np.zeros(n_samples)
        
        # Assign weights to each sample based on its class label
        for i, label in enumerate(y):
            sample_weights[i] = self.class_weights[label]

        for _ in range(self.epochs):
            # Compute logits and probabilities
            logits = np.dot(X, self.weights) + self.bias
            probabilities = self.sigmoid(logits)

            # Compute gradients with L2 regularisation and class weights
            dw = (1 / n_samples) * (np.dot(X.T, sample_weights * (probabilities - y)) + self.lambda_reg * self.weights)
            db = (1 / n_samples) * np.sum(sample_weights * (probabilities - y))

            # Update weights and bias
            self.weights = self.weights - self.eta * dw
            self.bias = self.bias - self.eta * db

    def compute_loss(self, X, y):
        """
        Compute the logistic loss with L2 regularisation and class weights.

        Parameters:
            X: Input data features
            y: Target labels 

        Returns:
            Loss value
        """
        logits = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(logits)
        
        # Compute weighted logistic loss
        weighted_log_loss = 0
        for i in range(len(y)):
            class_label = y[i] 
            weight = self.class_weights[class_label]  # Class weight
            
            # Compute negative log likelihood for a single sample
            log_prob = y[i] * np.log(probabilities[i] + 1e-10) + (1 - y[i]) * np.log(1 - probabilities[i] + 1e-10)
            
            # Add weighted log likelihood to the total loss
            weighted_log_loss += weight * log_prob 
        
        # Average the weighted log likelihood and add L2 regularisation term
        loss = -weighted_log_loss / len(y)
        l2_penalty = (self.lambda_reg / 2) * np.sum(self.weights**2)
        return loss + l2_penalty
    
    def clone(self):
        """
        Create a copy of the model.

        Returns:
            A copy of the model
        """
        return WeightedLogisticRegression(self.eta, self.epochs, self.lambda_reg, self.threshold, self.class_weights)