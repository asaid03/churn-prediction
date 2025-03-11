import numpy as np

class LogisticRegression:
    def __init__(self, n_epochs=1000):
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_epochs):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)
            self.weights -= dw
            self.bias -= db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)
        return [0 if y <= 0.5 else 1 for y in y_pred]

    def compute_loss(self, X, y):
        n_samples = X.shape[0]
        linear_pred = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_pred)
        loss = 0
        for i in range(n_samples):
            loss += -y[i] * np.log(predictions[i] + 1e-15) - (1 - y[i]) * np.log(1 - predictions[i] + 1e-15)
        return loss / n_samples

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))