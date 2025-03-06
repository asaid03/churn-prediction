import unittest
import numpy as np
from models.decision_tree import Node,DecisionTree 

class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        self.tree = DecisionTree(uniformity_measure="gini", max_depth=3)

    def test_impurity_gini(self):
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(self.tree.impurity(y), 0.5)

    def test_split(self):
        X_column = np.array([2, 3, 1, 5])
        left, right = self.tree.split(X_column, split_threshold=3)
        self.assertTrue(np.array_equal(left, [0, 1, 2])) 
        self.assertTrue(np.array_equal(right, [3]))       
        
    def test_split_empty_input(self):
        X_column = np.array([])
        left, right = self.tree.split(X_column, split_threshold=3)
        self.assertTrue(np.array_equal(left, []))  # Both left and right should be empty
        self.assertTrue(np.array_equal(right, []))
        
    def test_split_all_below_threshold(self):
        X_column = np.array([1, 2, 1, 0])
        left, right = self.tree.split(X_column, split_threshold=3)
        self.assertTrue(np.array_equal(left, [0, 1, 2, 3]))  
        self.assertTrue(np.array_equal(right, []))           


    def test_information_gain(self):
        labels = np.array([0, 0, 1, 1])
        feature_values = np.array([2, 3, 1, 5])
        gain = self.tree.information_gain(labels, feature_values, split_threshold=3)
        self.assertGreater(gain, 0)  # Gain should be positive

    def test_most_common_label(self):
        y = np.array([1, 0, 1, 1, 0])
        self.assertEqual(self.tree.most_common_label(y), 1)
            
    def test_clone(self):
        tree = self.tree.clone()
        self.assertNotEqual(tree, self.tree)
        self.assertEqual(tree.max_depth, self.tree.max_depth)
        self.assertEqual(tree.uniformity_measure, self.tree.uniformity_measure)
        self.assertEqual(tree.min_samples_split, self.tree.min_samples_split)
        self.assertEqual(tree.n_features, self.tree.n_features)

if __name__ == "__main__":
    unittest.main()
