from models.decision_tree import DecisionTree
import numpy as np
from collections import Counter

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
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._select_bootstrap_sample(X, y)
            tree = DecisionTree(
                uniformity_measure=self.uniformity_measure,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _select_bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        sample_idxs = np.random.choice(n_samples, n_samples, replace=True)
        X_sample = X[sample_idxs] 
        y_sample = y[sample_idxs]  
        return X_sample, y_sample

    def _majority_vote(self, preds):
        counter = Counter(preds)
        most_common_label = counter.most_common(1)[0][0]
        return  most_common_label
    
    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)  
        return np.array([self._majority_vote(preds) for preds in tree_predictions])
