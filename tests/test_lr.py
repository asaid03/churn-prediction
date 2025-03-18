import unittest
import numpy as np
from models.logistic_regression import LogisticRegression  

class TestLogisticRegression(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.model = LogisticRegression()
        self.X_train = np.random.rand(100, 3)
        self.y_train = np.random.randint(0, 2, size=self.X_train.shape[0])
        
    def test_initialisation(self):
        self.assertEqual(self.model.eta, 0.001)
        self.assertEqual(self.model.epochs, 1000)
        self.assertEqual(self.model.lambda_reg, 0.01)
        self.assertEqual(self.model.threshold, 0.5)
        self.assertIsNone(self.model.weights)
        self.assertIsNone(self.model.bias)
    
    def test_sigmoid_function(self):
        x = np.array([-100, 0, 100])
        expected = np.array([0, 0.5, 1])
        np.testing.assert_almost_equal(self.model.sigmoid(x), expected, decimal=2)
    
    def test_training(self):
        self.model.fit(self.X_train, self.y_train)
        self.assertIsNotNone(self.model.weights)
        self.assertIsNotNone(self.model.bias)
    
    def test_prediction(self):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_train)
        self.assertEqual(len(y_pred), len(self.y_train))
        self.assertTrue(all([y in [0, 1] for y in y_pred]))
    
    def test_loss_computation(self):
        self.model.fit(self.X_train, self.y_train)
        loss = self.model.compute_loss(self.X_train, self.y_train)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
    
    def test_clone(self):
        clone_model = self.model.clone()
        self.assertEqual(clone_model.eta, self.model.eta)
        self.assertEqual(clone_model.epochs, self.model.epochs)
        self.assertEqual(clone_model.lambda_reg, self.model.lambda_reg)
        self.assertEqual(clone_model.threshold, self.model.threshold)
        self.assertIsNone(clone_model.weights)
        self.assertIsNone(clone_model.bias)
        
    def test_all_zero_features(self)  :
        X = np.zeros((5, 3))  # All zero features
        y = np.array([0, 1, 0, 1, 0])  # Random labels

        model = LogisticRegression()
        model.fit(X, y)  # Should not break or produce NAN
        predictions = model.predict(X)
        
        assert (len(predictions) == 5, "Prediction length mismatch")
        
    def test_extreme_feature_values(self):
        X = np.array([[1000, -1000], [-1000, 1000], [5000, -5000]])  # ]Extreme feature values
        y = np.array([1, 0, 1]) 

        model = LogisticRegression()
        model.fit(X, y)  # Should not result in NaNs/infinity
        probabilities = model.predict_proba_per_sample(X)

        assert (np.all((probabilities >= 0) & (probabilities <= 1)), "Invalid probabilities detected")



if __name__ == "__main__":
    unittest.main()
