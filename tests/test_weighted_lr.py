import unittest
import numpy as np
from models.weighted_lr import WeightedLogisticRegression

class TestWeightedLogisticRegression(unittest.TestCase):

    def setUp(self):
        self.X_train = np.random.rand(100, 3)
        self.y_train = np.random.randint(0, 2, size=self.X_train.shape[0])

        self.class_weights = {0: 1, 1: 2}  #class 1 is twice as important
        self.model = WeightedLogisticRegression(eta=0.01, epochs=100, lambda_reg=0.1, class_weights=self.class_weights)

    def test_initialisation(self):
        self.assertEqual(self.model.eta, 0.01)
        self.assertEqual(self.model.epochs, 100)
        self.assertEqual(self.model.lambda_reg, 0.1)
        self.assertEqual(self.model.threshold, 0.5)
        self.assertEqual(self.model.class_weights, {0: 1, 1: 2})

    def test_fit(self):
        """Test if the model trains and updates weights properly."""
        self.model.fit(self.X_train, self.y_train)
        self.assertIsNotNone(self.model.weights)
        self.assertIsNotNone(self.model.bias)
        self.assertTrue(np.any(self.model.weights != 0))  # Weights should be updated

    def test_compute_loss(self):
        """Test if the loss calculated is a valid float and is positive."""
        self.model.fit(self.X_train, self.y_train)
        loss = self.model.compute_loss(self.X_train, self.y_train)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)  # Loss should be non-negative

    def test_clone(self):
        cloned_model = self.model.clone()
        self.assertIsInstance(cloned_model, WeightedLogisticRegression)
        self.assertEqual(cloned_model.eta, self.model.eta)
        self.assertEqual(cloned_model.epochs, self.model.epochs)
        self.assertEqual(cloned_model.lambda_reg, self.model.lambda_reg)
        self.assertEqual(cloned_model.threshold, self.model.threshold)
        self.assertEqual(cloned_model.class_weights, self.model.class_weights)
        
    def test_all_zero_features(self):
        X = np.zeros((5, 3))
        y = np.array([0, 0, 0, 0, 0])
        self.model.fit(X, y)
        predictions = self.model.predict(X)
        self.assertTrue(np.all(predictions == y)) # Predictions should be all zeros


if __name__ == '__main__':
    unittest.main()
