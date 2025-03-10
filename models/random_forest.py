from models.decision_tree import DecisionTree
import numpy as np
from collections import Counter
from joblib import Parallel, delayed  # Import parallel processing utilities

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, n_features=None, uniformity_measure="gini"):
        """
        Random Forest Classifier using Decision Trees.

        Parameters:
        - n_estimators: Number of decision trees.
        - max_depth: Maximum depth of each tree.
        - min_samples_split: Minimum samples required to split a node.
        - n_features: Number of features to consider when looking for the best split.
        - uniformity_measure: Criterion for measuring node uniformity ("gini", "entropy", "error").
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.uniformity_measure = uniformity_measure
        self.trees = []

    def fit(self, X, y):
        """
        Fit the random forest model to the training data.

        Parameters:
            X (numpy array): Training data features.
            y (numpy array): Training data labels.
        """
        # Parallelize the tree training
        # # n_jobs=-1 uses all available CPU cores
        self.trees = Parallel(n_jobs=-1)(  
            delayed(self.train_tree)(X, y) for _ in range(self.n_estimators)
        )

    def train_tree(self, X, y):
        """
        Helper function to train a single tree on a bootstrap sample.
        Will be parallelised by joblib.
        
        Parameters:
            X (numpy array): Training data features.
            y (numpy array): Training data labels.
            
        Returns:
            tree (DecisionTree): A trained decision tree.
        
        """
        tree = DecisionTree(
            uniformity_measure=self.uniformity_measure,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_features=self.n_features
        )
        X_sample, y_sample = self.select_bootstrap_sample(X, y)  # Bootstrap sampling
        tree.fit(X_sample, y_sample)  # Fit the tree on the bootstrap sample
        return tree

    def select_bootstrap_sample(self, X, y):
        """
        Generate a bootstrap sample from the dataset.

        Parameters:
            X (numpy array): Features of the input data.
            y (numpy array): Labels of the input data.
        
        Returns:
            X_sample (numpy array): Bootstrapped features.
            y_sample (numpy array): Bootstrapped labels.
        """
        n_samples = X.shape[0]
        # Randomly select indices with replacement
        sample_idxs = np.random.choice(n_samples, n_samples, replace=True)
        # bootstrap sample
        X_sample = X[sample_idxs] 
        y_sample = y[sample_idxs]  
        return X_sample, y_sample

    def majority_vote(self, preds):
        """
        Perform a majority vote on the predictions from the trees.

        Parameters:
            preds (list): List of predictions from the trees.

        Returns:
            most_common_label: The label that received the most votes.
        """
        counter = Counter(preds)
        most_common_label = counter.most_common(1)[0][0]
        return most_common_label

    def predict(self, X):
        """
        Predict the class labels for the input samples X.

        Parameters:
            X (numpy array): Input data features.

        Returns:
            predictions (numpy array): Predicted class labels.
        """
        # Collect predictions from each tree
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        # Majority vote for final prediction
        return np.array([self.majority_vote(preds) for preds in tree_predictions])

    def clone(self):
        """
        Create a new instance of RandomForest with the same parameters.

        Returns:
            RandomForest: A new instance of RandomForest with the same configuration.

        Note:
            This is mainly used for cross-validation.
        """
        return RandomForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_features=self.n_features,
            uniformity_measure=self.uniformity_measure
        )
