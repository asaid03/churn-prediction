import unittest
import numpy as np
import pickle
import os
from Evaluator.model_evaluator import ModelEvaluator


class TestModelEvaluator(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.y_true = np.array([1, 0, 1, 1, 0, 1, 0])  # True labels
        self.y_pred_correct = np.array([1, 0, 1, 1, 0, 1, 0])  # Perfect predictions
        self.y_pred_partial = np.array([1, 0, 0, 1, 1, 1, 0])  # Mixed predictions
        self.y_pred_wrong = np.array([0, 1, 0, 0, 1, 0, 1])  # Completely wrong

    def test_accuracy_score(self):
        self.assertAlmostEqual(ModelEvaluator.accuracy_score(self.y_true, self.y_pred_correct), 1.0, "Accuracy should be 1.0 for perfect predictions.")
        self.assertAlmostEqual(ModelEvaluator.accuracy_score(self.y_true, self.y_pred_partial), 5/7, "Accuracy should match the correct predictions ratio.")
        self.assertAlmostEqual(ModelEvaluator.accuracy_score(self.y_true, self.y_pred_wrong), 0.0, "Accuracy should be 0.0 for completely wrong predictions.")

    def test_precision_score(self):
        self.assertAlmostEqual(ModelEvaluator.precision_score(self.y_true, self.y_pred_correct), 1.0, "Precision should be 1.0 for perfect predictions.")
        self.assertAlmostEqual(ModelEvaluator.precision_score(self.y_true, self.y_pred_partial), 0.75, places=4, msg="Precision calculation is incorrect.")
        self.assertAlmostEqual(ModelEvaluator.precision_score(self.y_true, self.y_pred_wrong), 0.0, "Precision should be 0.0 when no positives are predicted correctly.")

    def test_recall_score(self):
        self.assertAlmostEqual(ModelEvaluator.recall_score(self.y_true, self.y_pred_correct), 1.0, "Recall should be 1.0 for perfect predictions.")
        self.assertAlmostEqual(ModelEvaluator.recall_score(self.y_true, self.y_pred_partial), 0.75, places=4, msg="Recall calculation is incorrect.")
        self.assertAlmostEqual(ModelEvaluator.recall_score(self.y_true, self.y_pred_wrong), 0.0, "Recall should be 0.0 when no true positives are predicted.")

    def test_f1_score(self):
        self.assertAlmostEqual(ModelEvaluator.f1_score(self.y_true, self.y_pred_correct), 1.0, "F1-Score should be 1.0 for perfect predictions.")
        self.assertAlmostEqual(ModelEvaluator.f1_score(self.y_true, self.y_pred_partial), 0.75, places=4, msg="F1-Score calculation is incorrect.")
        self.assertAlmostEqual(ModelEvaluator.f1_score(self.y_true, self.y_pred_wrong), 0.0, "F1-Score should be 0.0 when precision and recall are both 0.")

    def test_f2_score(self):
        self.assertAlmostEqual(ModelEvaluator.f2_score(self.y_true, self.y_pred_correct), 1.0, "F2-Score should be 1.0 for perfect predictions.")
        self.assertAlmostEqual(ModelEvaluator.f2_score(self.y_true, self.y_pred_partial), 0.75, places=4, msg="F2-Score calculation is incorrect.")
        self.assertAlmostEqual(ModelEvaluator.f2_score(self.y_true, self.y_pred_wrong), 0.0, "F2-Score should be 0.0 when precision and recall are both 0.")

    def test_error_handling(self):
        # Test shape mismatch
        with self.assertRaises(ValueError):
            ModelEvaluator.accuracy_score(self.y_true, self.y_pred_correct[:-1])

        with self.assertRaises(ValueError):
            ModelEvaluator.precision_score(self.y_true, self.y_pred_correct[:-1])

        with self.assertRaises(ValueError):
            ModelEvaluator.recall_score(self.y_true, self.y_pred_correct[:-1])


if __name__ == "__main__":
    unittest.main()

