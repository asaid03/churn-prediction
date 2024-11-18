import copy
import heapq
import math
import numpy as np


class NearestNeighbours:
    
    def __init__(self,neighbours=1):
        
        if neighbours < 1:
            self.n_neighbours = 1
        else:
            self.n_neighbours = neighbours
        self.tie_count = 0  # Initialising the tie counter
        
    
            
    
    def get_Num_Neighbours(self):
        return self.n_neighbours
    
    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    
    def euclidean_distance(self,p1,p2):
        return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))
    

    def get_nearest_neighbour(self, test_sample):
        # Checks if we set K > size of sample and deals with it accordingly
        k = min(self.n_neighbours, len(self.X_train))

        # Calculate all distances without sorting
        distances = [
            (self.euclidean_distance(train_sample, test_sample), label)
            for train_sample, label in zip(self.X_train, self.y_train)
        ]

        # Use heapq to get the k smallest distances directly
        k_nearest_neighbours = heapq.nsmallest(k, distances, key=lambda x: x[0])
        return k_nearest_neighbours



    def vote_label(self,distances): #vote label by returning label with the most counts.
        labels = []
        for distance,label in distances:
            labels.append(label)
            
        unique_labels, counts = np.unique(labels, return_counts=True)
        max_count = np.max(counts)
        max_labels = unique_labels[counts == max_count]

        # Update tie counter if there's a tie
        if len(max_labels) > 1:
            self.tie_count += 1  # Increment tie counter when there's a tie

        # Return the first label in case of a tie or the most frequent label
        index = np.argmax(counts)
        return unique_labels[index]

    def predict(self, X_test):
        predictions = []

        for i in X_test:
            nearest_neighbours = self.get_nearest_neighbour(i)
            label = self.vote_label(nearest_neighbours)
            predictions.append(label)
            
    
        return predictions
    
    
    def score(self,predictions,y_test):
        error_rate = 1 -np.mean(predictions == y_test)
        num_errors = np.size(predictions) - np.count_nonzero(predictions == y_test)
        print("Error rate is: " + str(error_rate))
        print("Number of errors: " + str(num_errors) + " out of " + str(len(y_test)) )
        
    def error_rate(self,predictions,y_test):
        error_rate = 1 -np.mean(predictions == y_test)
        return error_rate
    
    def print_tie_count(self):
        print(f"Number of ties encountered: {self.tie_count}")
        
        
        
        
        
