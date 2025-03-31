import numpy as np
from models.knn import NearestNeighbours

class Smote:
    """
    Synthetic Minority Over-sampling Technique (SMOTE) for imbalanced classification problems.
    It works by generating synthetic samples for the minority class to balance the dataset.
    To generate the synthetic samples:
    1. Randomly select a sample from the minority class
    2. Find its k-nearest neighbours
    3. Randomly select one of these neighbours
    4. Generate a synthetic sample by interpolating between the original sample and the chosen neighbour
    
    """
    def __init__(self, k_neighbours=5, random_state=None):
        
        """
        Initialises a SMOTE object.
        
        Parameters:
        - k_neighbours: Number of nearest neighbours to consider when generating synthetic samples . we use a default of k_neighbours=5
        - random_state: Seed for random number generation 
        
        """
        self.k_neighbours = k_neighbours
        self.random_state = random_state
        
        
        
    def find_minority_class(self, y):
        """
        It finds the minority class in the target labels.
        
        Parameters:
        - y: Target labels
        
        Returns:
        - minority_churn_class: The minority class label
        
        """
        churn_counts = np.bincount(y)  # Count occurrences of each label
        minority_churn_class = np.argmin(churn_counts)  # Find the label with the lowest count
        return minority_churn_class
        

    def fit_resample(self, X, y):
        """
        Generate synthetic samples for the minority class and balance the dataset.
        
        Parameters:
        - X: Features
        - y: Target labels
        
        Returns:
        - X_resampled: Resampled features
        - y_resampled: Resampled target labels
        
        """
        np.random.seed(self.random_state)

        minority_class = self.find_minority_class(y)
        X_minority = X[y == minority_class]

        num_minority = len(X_minority)
        num_majority = len(y) - num_minority  # Compute majority class size

        # Generate enough samples to fully balance classes
        num_synthetic = num_majority - num_minority

        if num_synthetic <= 0:
            return X, y  # Already balanced

        knn = NearestNeighbours(neighbours=self.k_neighbours)
        knn.fit(X_minority, np.arange(num_minority))

        synthetic_samples = []
        for _ in range(num_synthetic):
            random_index = np.random.randint(num_minority) # Randomly select a sample from the minority class
            original_sample = X_minority[random_index] # Get the original sample

            neighbours = knn.get_nearest_neighbour(original_sample)
            possible_indices = [neighbour[1] for neighbour in neighbours]
            
            # Randomly select a neighbour from the k-nearest neighbours
            chosen_neighbour_idx = np.random.choice(possible_indices) 
            chosen_neighbour = X_minority[chosen_neighbour_idx]
            
            # Generate a synthetic sample by interpolating between the original minority sample and the chosen neighbour
            interpolation_factor = np.random.random() # Randomly select an interpolation factor
            synthetic_sample = original_sample + interpolation_factor * (chosen_neighbour - original_sample) # Generate synthetic sample using this equation

            synthetic_samples.append(synthetic_sample)

        # Combine the original dataset with the synthetic samples
        synthetic_samples = np.array(synthetic_samples)
        X_resampled = np.vstack((X, synthetic_samples))
        
        synthetic_labels = np.full(num_synthetic, minority_class) # Generate labels for the synthetic samples
        y_resampled = np.hstack((y, synthetic_labels))

        return X_resampled, y_resampled
    
    
