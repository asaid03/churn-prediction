import unittest
import numpy as np
from training.crossval import CrossValidator
from models.decision_tree import DecisionTree

class TestCrossValidatorWithDecisionTree(unittest.TestCase):
    def setUp(self):
        self.tree = DecisionTree(
            uniformity_measure="gini",
            max_depth=3,
            min_samples_split=2,
            n_features=None
        )

        # Generate a small dummy dataset
        np.random.seed(42)
        self.X_train = np.random.rand(20, 5)
        self.y_train = np.random.randint(0, 2, 20)

    def test_cross_validate_runs_without_errors(self):
        results = CrossValidator.cross_validate(self.tree, self.X_train, self.y_train, folds=5)
        # check that it returns the expected keys
        self.assertIn("mean_metrics", results)
        self.assertIn("std_metrics", results)
        self.assertIn("fold_metrics", results)

    def test_cross_validate_returns_correct_number_of_folds(self):
        fold_sizes = [3, 5, 10]
        for folds in fold_sizes:
            with self.subTest(folds=folds):  # Test multiple fold sizes
                results = CrossValidator.cross_validate(self.tree, self.X_train, self.y_train, folds=folds)
                self.assertEqual(len(results["fold_metrics"]), folds)


    def test_cross_validate_metrics_are_valid(self):
        # Run cross-validation
        results = CrossValidator.cross_validate(self.tree, self.X_train, self.y_train, folds=5)
        expected_metrics = ["accuracy", "precision", "recall", "f1_score", "f2_score"]

        for metric in expected_metrics:
            with self.subTest(metric=metric): 
                # Ensure the metric exists in the mean and std dictionaries
                self.assertIn(metric, results["mean_metrics"], f"{metric} missing")
                self.assertIn(metric, results["std_metrics"], f"{metric} missing")

                # Ensure the metric values are within [0,1]
                self.assertTrue(0 <= results["mean_metrics"][metric] <= 1, f"{metric} mean out of range")
                self.assertTrue(0 <= results["std_metrics"][metric] <= 1, f"{metric} std out of range")


if __name__ == "__main__":
    unittest.main()
