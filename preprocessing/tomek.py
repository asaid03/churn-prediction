import numpy as np
from models.knn import NearestNeighbours

class TomekLinks:

    def __init__(self, k=1):
        self.k = k

    def fit_resample(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        tomek_links = self.get_tomek_links(X, y)
        majority_class = self.majority_class(y)
        remove_samples = set()
        
        # Identify majority class samples in Tomek Links
        for i, j in tomek_links: 
            if y[i] == majority_class:
                remove_samples.add(i) 
            if y[j] == majority_class:
                remove_samples.add(j)
        
        #remove samples from the dataset
        remove_indices = list(remove_samples)
        X_resampled = np.delete(X, remove_indices, axis=0)
        y_resampled = np.delete(y, remove_indices, axis=0)

        return X_resampled, y_resampled

    def get_tomek_links(self, X, y):

        tomek_links = set()
        n_samples = len(X)

        # Find the nearest neighbour of each sample
        for current_idx in range(n_samples):
            nn_idx = self.get_cloesest_neighbour_index(X, current_idx)

            if nn_idx is not None:
                reverse_nn_idx = self.get_cloesest_neighbour_index(X, nn_idx)

                if reverse_nn_idx == current_idx and y[current_idx] != y[nn_idx]: #check if nearest neighbour is the same i.e its mutual
                    # store in ascending order to avoid duplicates
                    link = (min(current_idx, nn_idx), max(current_idx, nn_idx))
                    tomek_links.add(link)

        return tomek_links

    def get_cloesest_neighbour_index(self, X,index):
        min_dist = float('inf')
        min_index = None
        for samples in range(len(X)):
            if samples == index:
                continue
            dist = self.euclidean_distance(X[index],X[samples])
            if dist < min_dist:
                min_dist = dist
                min_index = samples
        return min_index        
    
    def euclidean_distance(self, s1, s2):
        """
        Compute the Euclidean distance between two samples.
        
        Parameters:
        - x1: First sample
        - x2: Second sample
        
        Returns:
        - distance: Euclidean distance between x1 and x2
        
        """
        return np.sqrt(np.sum((s1 - s2) ** 2))
    
    
    def majority_class(self, y):
        """
        Find the majority class in the target labels.
        
        Parameters:
        - y: Target labels
        
        Returns:
        - majority_class: The majority class label
        
        """
        churn_counts = np.bincount(y)  # Count occurrences of each label
        majority_class = np.argmax(churn_counts)  # Find the label with the highest count
        return majority_class