import numpy as np
from evaluator.model_evaluator import ModelEvaluator

class CrossValidator:
    @staticmethod
    def cross_validate(model, X_train, y_train, folds=5, random_state=None):
        """
        Cross validate a model
        :param model: Model to cross validate 
        :param X_train: Training data
        :param y_train: Training labels
        :param folds: Number of folds
        :param random_state: Random state for reproducibility  
        """
        sample_indexes = np.arange(X_train.shape[0])
        if random_state is not None :
            np.random.seed(random_state)
            np.random.shuffle(sample_indexes)

        fold_size = len(X_train) // folds
        model_performance = []

        for i in range(folds):
            # Split sample_indexes
            val_start = i * fold_size
            val_end = (i + 1) * fold_size
            val_idx = sample_indexes[val_start:val_end]
            train_idx = np.concatenate([sample_indexes[:val_start], sample_indexes[val_end:]])

            # Split data
            X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
            X_fold_val, y_fold_val = X_train[val_idx], y_train[val_idx]

            # Clone and train model
            fold_model = model.clone() 
            fold_model.fit(X_fold_train, y_fold_train)

