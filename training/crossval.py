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

        for fold in range(folds):
            # validation set indexes
            validation_start = fold * fold_size
            validation_end = (fold + 1) * fold_size if fold < folds - 1 else len(X_train)

            validation_indexes = sample_indexes[validation_start:validation_end] # range of indexes for validation set from sample_indexes to be used for validation set
            # training set indexes
            training_indexes_start = sample_indexes[:validation_start]
            training_indexes_end = sample_indexes[validation_end:]
            training_indexes = np.concatenate([training_indexes_start, training_indexes_end])

            # Split training and validation sets
            X_fold_train, y_fold_train = X_train[training_indexes], y_train[training_indexes]
            X_fold_validation, y_fold_validation = X_train[validation_indexes], y_train[validation_indexes]

            # Clone model for each fold
            fold_model = model.clone()
            fold_model.fit(X_fold_train, y_fold_train)

            # Predict/evaluate
            y_pred = fold_model.predict(X_fold_validation)
            model_performance.append(ModelEvaluator.calculate_metrics(y_fold_validation, y_pred))

        # metrics for each fold
        mean_metrics = {}
        std_metrics = {}

        for metric in model_performance[0]:
            metric_values = [fold_metrics[metric] for fold_metrics in model_performance]
            mean_metrics[metric] = np.mean(metric_values)
            std_metrics[metric] = np.std(metric_values)

        return {
            'mean_metrics': mean_metrics,
            'std_metrics': std_metrics,
            'fold_metrics': model_performance
        }



