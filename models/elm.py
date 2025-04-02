import numpy as np

class ELM:
    """
    Extreme Learning Machine.

    """
    def __init__(self, hidden_nodes=100, activation='sigmoid', random_state=None):
        self.hidden_nodes = hidden_nodes
        self.activation = activation
        self.random_state = random_state
        self.input_weights = None
        self.biases = None
        self.output_weights = None

    def activation_function(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")


    def hidden_layer_output(self, X):

        X = X.T
        G = np.dot(self.input_weights, X) + self.biases
        
        return self.activation_function(G)

    def fit(self, X, y):
        n_features = X.shape[1]
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialise input weights and biases
        self.input_weights = np.random.randn(self.hidden_nodes, n_features)
        self.biases = np.random.randn(self.hidden_nodes, 1)

        y = y.reshape(1, -1)
        H = self.hidden_layer_output(X)
        H_T = H.T
        
        
        self.output_weights = np.dot(np.linalg.pinv(H_T), y.T)

    def predict(self, X):

        H = self.hidden_layer_output(X)
        raw_output = np.dot(self.output_weights.T, H)
        
        #convert into binary_classification output
        return (raw_output > 0.5).astype(int).ravel()

