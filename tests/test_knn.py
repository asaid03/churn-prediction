import unittest
from models.knn import NearestNeighbours,Conformal
import unittest
import numpy as np


class TestNearestNeighbours(unittest.TestCase):

    def setUp(self):
        # Create a simple dataset
        self.X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 7]])
        self.y_train = np.array([0, 0, 1, 1])
        self.X_test = np.array([[1, 2], [6, 6]])
        self.y_test = np.array([0, 1])
        
        self.knn = NearestNeighbours(neighbours=2)
        self.knn.fit(self.X_train, self.y_train)

    def test_euclidean_distance(self):
        dist = self.knn.euclidean_distance([0, 0], [3, 4])
        self.assertAlmostEqual(dist, 5.0, "Euclidean distance calculation is incorrect.")

    def test_get_nearest_neighbour(self):
        nearest = self.knn.get_nearest_neighbour([1, 2])
        self.assertEqual(len(nearest), 2, "Number of neighbours retrieved is incorrect.")
        self.assertEqual(nearest[0][1], 0, "Nearest neighbour label is incorrect.")

    def test_predict(self):
        predictions = self.knn.predict(self.X_test)
        self.assertEqual(predictions, [0, 1], "Prediction is incorrect.")

    def test_score(self):
        predictions = self.knn.predict(self.X_test)
        error_rate = self.knn.error_rate(predictions, self.y_test)
        self.assertAlmostEqual(error_rate, 0.0, "Error rate calculation is incorrect.")

class TestConformal(unittest.TestCase):

    def setUp(self):
        # Create a dataset
        self.X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 7]])
        self.y_train = np.array([0, 0, 1, 1])
        self.X_test = np.array([[1, 2], [6, 6]])
        self.y_test = np.array([0, 1])

        self.conformal = Conformal()
        self.conformal.fit(self.X_train, self.y_train)
        self.conformal.str_test(self.X_test, self.y_test)

    def test_conformity_score(self):
        score = self.conformal.conformity_score(5, 3)
        self.assertAlmostEqual(score, 5 / 3, "Conformity score calculation is incorrect.")

    def test_get_possible_labels(self):
        possible_labels = self.conformal.get_possible_labels(self.y_train)
        self.assertEqual(possible_labels, [0, 1], "Possible labels retrieval is incorrect.")


    def test_pvalue(self):
        sample = np.append(self.X_test[0], self.y_test[0])
        cs_list = self.conformal.cs_test_sample(sample)
        pval = self.conformal.pvalue(sample, cs_list)
        self.assertTrue(0 <= pval <= 1, "P-value is not in the valid range.")

    def test_avg_false_pval_1sample(self):
        sample = self.X_test[0]
        avg_false_pval = self.conformal.avg_false_pval_1sample(sample)
        self.assertTrue(0 <= avg_false_pval <= 1, "Average false p-value is not in the valid range.")

    def test_avg_false_pval(self):
        avg_false_pval = self.conformal.avg_false_pval()
        self.assertTrue(0 <= avg_false_pval <= 1, "Average false p-value for test set is not in the valid range.")


if __name__ == '__main__':
    unittest.main()

