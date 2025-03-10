import unittest
import numpy as np
from models.decision_tree import DecisionTree
from models.random_forest import RandomForest

class TestRandomForest(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)  
        self.X_train = np.random.rand(100, 5)  
        self.y_train = np.random.choice([0, 1], size=100)  
        self.forest = RandomForest(n_estimators=5, max_depth=3)

    def test_initialisation(self):
        self.assertEqual(self.forest.n_estimators, 5)
        self.assertEqual(self.forest.max_depth, 3)
        self.assertEqual(len(self.forest.trees), 0)
        self.assertIn(self.forest.uniformity_measure, ["gini", "entropy" , "error"]) 

    def test_fit(self):
        self.forest.fit(self.X_train, self.y_train)
        self.assertGreater(len(self.forest.trees), 0) 
        self.assertIsInstance(self.forest.trees[0], DecisionTree) # Check type of first tree is a DecisionTree

    def test_predict(self):
        self.forest.fit(self.X_train, self.y_train)
        predictions = self.forest.predict(self.X_train)
        self.assertEqual(len(predictions), len(self.y_train)) # Check length of predictions matches with y_train
        self.assertTrue(set(predictions).issubset({0, 1}))  # Predictions should be 0 or 1

    def test_clone(self):
        new_forest = self.forest.clone()
        self.assertIsInstance(new_forest, RandomForest)
        self.assertEqual(new_forest.n_estimators, self.forest.n_estimators)
        self.assertEqual(len(new_forest.trees), 0) 

    def test_bootstrap_sample(self):
        X_sample, y_sample = self.forest.select_bootstrap_sample(self.X_train, self.y_train)
        self.assertEqual(X_sample.shape, self.X_train.shape) # Check shape of X_sample matches X_train
        self.assertEqual(y_sample.shape, self.y_train.shape) # check shape of y_sample matches y_train

if __name__ == "__main__":
    unittest.main()
