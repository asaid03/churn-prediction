import unittest
import numpy as np
from models.elm import ELM  


class TestELM(unittest.TestCase):

    def test_activation_functions(self):
        x = np.array([[0, 1, -1]])

        sigmoid = ELM(activation='sigmoid')
        expected_sigmoid = 1 / (1 + np.exp(-x))
        self.assertTrue(np.allclose(sigmoid.activation_function(x), expected_sigmoid))

        tanh = ELM(activation='tanh')
        expected_tanh = np.tanh(x)
        self.assertTrue(np.allclose(tanh.activation_function(x), expected_tanh))

        relu = ELM(activation='relu')
        expected_relu = np.maximum(0, x)
        self.assertTrue(np.array_equal(relu.activation_function(x), expected_relu))

    def test_invalid_activation_raises(self):  
        model = ELM(activation='sideye..kk')
        
        with self.assertRaises(ValueError):
            model.activation_function(np.array([[-1.0001]]))


    def test_if_pred_test_shape_matches(self):
        X = np.array([[0], [1]])
        y = np.array([0, 1])
        model = ELM(hidden_nodes=5, random_state=42)
        model.fit(X, y)
        
        preds = model.predict(X)

        self.assertEqual(preds.shape, y.shape)
        self.assertTrue(np.all(np.isin(preds, [0, 1])))

    def test_random_seed_results_matches(self):
        X = np.array([[0], [1]])
        y = np.array([0, 1])
        
        model1 = ELM(hidden_nodes=5, random_state=1)
        model2 = ELM(hidden_nodes=5, random_state=1)
        
        model1.fit(X, y)
        model2.fit(X, y)

        self.assertTrue(np.array_equal(model1.predict(X), model2.predict(X)))


if __name__ == '__main__':
    unittest.main()
