import pandas as pd
import numpy as np
from collections import Counter
import pickle
import os

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        
        """
        Initialises a node in the decision tree.

        Parameters:
        - feature: Index of the feature used for splitting.
        - threshold: Threshold value for the split.
        - left: Left child node.
        - right: Right child node.
        - value: Class label if the node is a leaf.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, uniformity_measure="gini", max_depth=10, min_samples_split=2, n_features=None):
        """
        Decision Tree Classifier supporting Gini, Entropy, and Classification Error.

        Parameters:
        - uniformity_measure: ('gini', 'entropy', 'error')
        - max_depth: Maximum depth of the tree.
        - min_samples_split: Minimum number of samples required to split a node.
        - n_features: Number of features to consider for splits (uses all if None).
        """
        self.uniformity_measure = uniformity_measure
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None
        
        valid_uniformity_measures = {"gini", "entropy", "error"}
        if self.uniformity_measure not in valid_uniformity_measures:
            raise ValueError(f"Invalid uniformity_measure {self.uniformity_measure}. Choose from {valid_uniformity_measures}.")
        
        # Validate max_depth
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth <= 0):
            raise ValueError("max_depth must be a positive integer or None.")


        # Validate min_samples_split
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2.")

        # Validate n_features
        if n_features is not None:
            if not isinstance(n_features, int) or n_features <= 0:
                raise ValueError("n_features must be a positive integer or None.")
        
        
        
    def fit(self, X, y):
        """
        Build the decision tree classifier from the training set (X, y).

        Parameters:
        - X: Feature
        - y: Label 
        """

        # decide number of features for  splitting
        if not self.n_features:
            # if not specified use all freatures available
            current_n_features = X.shape[1]
        else:
            # case for n_features  being smaller then actual size of features.
            current_n_features = min(X.shape[1], current_n_features)

        self.root = self.grow_tree(X, y,current_n_features)

    def predict(self, X):
        """
        Predict class labels for the input samples.
        parameters:
        - X: Input samples.
        Returns the predicted class labels.
        """
        return np.array([self.traverse_tree(x, self.root) for x in X])

    def grow_tree(self, X, y,current_n_features, depth=0):
        """
        Recursively build the decision tree.
        parameters:
        - X: Feature values.
        - y: Labels of the samples.
        - current_n_features: Number of features to consider for splitting.
        - depth: Current depth of the tree.
        Returns the root node of the tree.
        """
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if ((self.max_depth is not None and depth >= self.max_depth)
            or n_labels == 1
            or n_samples < self.min_samples_split):
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        # Select features to consider
        feat_idxs = np.random.choice(n_feats,current_n_features, replace=False)

        # Find the best split
        best_feature, best_threshold = self.best_split(X, y, feat_idxs)

        # Create child nodes
        left_idxs, right_idxs = self.split(X[:, best_feature], best_threshold)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:  # Handle edge case of empty splits
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        left_child = self.grow_tree(X[left_idxs, :], y[left_idxs],current_n_features, depth + 1)
        right_child = self.grow_tree(X[right_idxs, :], y[right_idxs],current_n_features, depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def best_split(self, X, y, feat_idxs):
        """
        Find the best feature and threshold for splitting the data.
        parameters:
        - X: Feature values.
        - y: Labels of the samples.
        - feat_idxs: Indices of the features to consider for splitting.
        Returns the index of the best feature and the best threshold for splitting.
        """
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # Calculate the information gain
                gain = self.information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold
    

    def information_gain(self, labels, feature_values, split_threshold):
        """
        Calculate the information gain from a split.
        parameters:
        - labels: Labels of the samples.
        - feature_values: Feature values for the current feature.
        - split_threshold: Threshold value for splitting.
        Returns the information gain from the split.
        """

        # Impurity of the parent node before the splitting
        parent_impurity = self.impurity(labels)

        # Split data based on the threshold
        left_indices, right_indices = self.split(feature_values, split_threshold)
        if len(left_indices) == 0 or len(right_indices) == 0:  # Handle edge cases where no split occurs
            return 0

        # Calculate the weighted average impurity of the child nodes
        total_samples = len(labels)
        left_size, right_size = len(left_indices), len(right_indices)
        left_impurity = self.impurity(labels[left_indices])
        right_impurity = self.impurity(labels[right_indices])
        weighted_child_impurity = (left_size / total_samples) * left_impurity + (right_size / total_samples) * right_impurity

        # Information gain is the reduction in impurity
        information_gain = parent_impurity - weighted_child_impurity
        return information_gain


    def split(self, X_column, split_threshold):
        """
        parameters:
        - X_column: Feature values for the current feature.
        - split_threshold: Threshold value for splitting.
        Returns the indices of the samples in the left and right child nodes.
        
        """
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        return left_idxs, right_idxs

    def impurity(self, y):
        """
       parameters:
        - y: Labels of the samples.
        Returns the impurity of the labels using the specified measure.
        """
        proportions = np.bincount(y) / len(y) # p(X)

        if self.uniformity_measure == "gini":
            return 1 - np.sum(proportions ** 2)
        elif self.uniformity_measure == "entropy":
            return -np.sum([p * np.log2(p) for p in proportions if p > 0])
        elif self.uniformity_measure == "error":
            return 1 - np.max(proportions)
        

    def most_common_label(self, y):
        """
        Find the most common label in the labels array.
        parameters:
        - y: Labels of the samples.
        Returns the most common label.
        
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0] # returns the label with the most counts
    

    def traverse_tree(self, x, node):
        """
        Traverse the tree to find the predicted class for a sample.
        parameters:
        - x: Input sample.
        - node: Current node in the tree.
        Returns the predicted class label.
        
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)
    
    def clone(self):
        '''
        Create a clone of the current DecisionTree instance.
        Used by my cross-validation class.
        Returns:
        - A new DecisionTree instance with the same parameters.
        '''
        return DecisionTree(
            uniformity_measure = self.uniformity_measure,
            max_depth = self.max_depth,
            min_samples_split = self.min_samples_split,
            n_features = self.n_features
        )


    def save(self, filename):
        """
        Save the trained model to a file.

        Parameters:
        - filename: Name of the file to save the model.
        """
        os.makedirs("checkpoints", exist_ok=True)
        file_path = os.path.join("checkpoints", filename)
        
        # Save the model
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {file_path}")

    @staticmethod
    def load(filename):
        """
        Load a trained model from a file.

        Parameters:
        - filename: Name of the file containing the saved model.

        Returns:
        - model: The loaded DecisionTree model.
        """
        # Construct the full file path
        file_path = os.path.join("checkpoints", filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load the model
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {file_path}")
        return model




