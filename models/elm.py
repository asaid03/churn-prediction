import numpy as np

class ELM:
    """
    Extreme Learning Machine.
    
    An implementation of a single hidden layer feedforward neural network for binary classification.
    It uses a randomised approach to initialise the input weights and biases
    This is then used to compute the output weights using the Moore Penrose pseudoinverse.

    """
    def __init__(self, hidden_nodes=100, activation='sigmoid', random_state=None):
        """
        Initialises the ELM model.
        
        Parameters:
            hidden_nodes (int): Number of hidden nodes in the hidden layer.
                                The default is 100. 
            activation (str): Activation function to use in the hidden layer. Options are 'sigmoid', 'tanh', and 'relu'. 
                              The default is sigmoid. 
            random_state (int): Random seed for reproducibility. The default is None.
        """
        
        self.hidden_nodes = hidden_nodes
        self.activation = activation
        self.random_state = random_state
        self.input_weights = None
        self.biases = None
        self.output_weights = None

    def activation_function(self, x):
        """
        Applies the specified activation function to the input.
        This function is used in the hidden layer of the ELM.
        
        The possible activation functions are:
        - sigmoid: Sigmoid activation function.   
        - tanh: Hyperbolic tangent activation function.
        - relu: Rectified Linear Unit activation function.
        
        parameters:
            x: Input data.
            
        returns: An output after applying the activation function.
        
        """
        
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")


    def hidden_layer_output(self, X):
        """
        Computes the output of the hidden layer.
        
        Parameters:
            X: Input data.
        
        returns: Output of the hidden layer after applying the activation function.
        
        """

        X = X.T # Transpose for matrix multiplication
        
        G = np.dot(self.input_weights, X) + self.biases # pre activation output:  weights * inputs + biases
        
        return self.activation_function(G) # hidden layer output with activation function

    def fit(self, X, y):
        """
        The training function for the ELM model.
        
        parameters:
            X: Input data.
            y: Target labels.
        
        returns: None.
        """
        n_features = X.shape[1]
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # randomly initialise input weights and biases
        self.input_weights = np.random.randn(self.hidden_nodes, n_features)
        self.biases = np.random.randn(self.hidden_nodes, 1)

        y = y.reshape(1, -1) # we do this for matrix multiplication
        
        #compute hidden layer output
        H = self.hidden_layer_output(X)
        H_T = H.T # Transpose for matrix multiplication
        
        # computes the output weights.
        # we do this using lianlg.pinv a numpy func  that computes the pseudo-inverse of a matrix.
        self.output_weights = np.dot(np.linalg.pinv(H_T), y.T)

    def predict(self, X):
        
        """
        This is the prediction function for the ELM model
        
        parameters:
            X: Input data.
        
        returns: a binary classification output of 0 or 1
        """

        H = self.hidden_layer_output(X)
        raw_output = np.dot(self.output_weights.T, H)
        
        #convert into binary_classification output i.e 0 or 1
        return (raw_output > 0.5).astype(int).ravel()

