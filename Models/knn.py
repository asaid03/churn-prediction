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
        

class Conformal(NearestNeighbours):
    
    def __init__(self,):
        super().__init__(neighbours=1)
        self.training_set_cs = None
        self.pos_labels = None

        
    def fit(self,X_train,y_train):
        super().fit(X_train,y_train)
        self.training_set_cs = self.cs_train_set() # call cs_train_set so we can use later for cs of test sample
        self.pos_labels = self.get_possible_labels(y_train)
        
    def str_test(self,X_test,y_test):
        self.X_test = X_test
        self.y_test = y_test

    def get_possible_labels(self,training_labels):
        possible_labels = set(training_labels)
        return list(possible_labels)

    def conformity_score(self, near_diff_distance, near_same_distance):

        if near_same_distance == 0 and near_diff_distance > 0:
            return np.inf
        elif near_same_distance == 0 and near_diff_distance == 0:
            return 0
        elif near_same_distance != 0:
            return near_diff_distance / near_same_distance


    def find_minimum(self, distances):
        if not distances:
            return None  
        minimum = distances[0]
        for dist in distances:
            if dist < minimum:
                minimum = dist
        return minimum

    def cs_train_set(self):
        cs = []
        for i in range(len(self.X_train)):
            near_diff_distance = []  # distances to different class samples
            near_same_distance = []  # distances to same class samples
            sample = self.X_train[i]
            labelled_sample = np.append(sample, self.y_train[i]) 

            for j in range(len(self.X_train)):
                if i != j:  # skip the current sample itself
                    dist = self.euclidean_distance(self.X_train[i], self.X_train[j])

                    if self.y_train[i] == self.y_train[j]:
                        near_same_distance.append(dist)
                    else:
                        near_diff_distance.append(dist)

            min_diff_distance = self.find_minimum(near_diff_distance) 
            min_same_distance = self.find_minimum(near_same_distance)
            # Compute conformity score for the current sample
            conformity_score = self.conformity_score(min_diff_distance, min_same_distance)
            
            row = []
            row.append(conformity_score)
            row.append(labelled_sample)
            row.append(min_diff_distance)
            row.append(min_same_distance)
            cs.append(row)


        return cs

                   
           
    def cs_test_sample(self, sample): # reduce computation by only computing c.s for sample if there is change in distance of nearest difference class and nearest same class
        test_sample = sample[:-1]
        test_label = sample[-1]

        train_set_conformity_scores = copy.deepcopy(self.training_set_cs)

        near_diff_distance = math.inf
        near_same_distance = math.inf
        
        changed = False

        for i in range(len(train_set_conformity_scores)):
            train_sample = train_set_conformity_scores[i][1][:-1]
            train_label = train_set_conformity_scores[i][1][-1]
            min_diff_distance = train_set_conformity_scores[i][2]
            min_same_distance = train_set_conformity_scores[i][3]
            
                        
            dist = self.euclidean_distance(train_sample, test_sample)

            if (train_label == test_label):
                if (dist <= near_same_distance):
                    near_same_distance = dist
                    
                if(dist <= min_same_distance):
                    train_set_conformity_scores[i][3] = dist
                    changed = True
                         
            else:
                if (dist <= near_diff_distance):
                    near_diff_distance = dist
                    
                if(dist<= min_diff_distance):
                    train_set_conformity_scores[i][2] = dist
                    changed = True
                
            if changed:
                train_set_conformity_scores[i][0] = self.conformity_score(train_set_conformity_scores[i][2], train_set_conformity_scores[i][3])
                changed = False
        
         
        conformity_score = self.conformity_score(near_diff_distance, near_same_distance)
        row = []
        row.append(conformity_score)
        row.append(sample)
        row.append(near_diff_distance)
        row.append(near_same_distance)
        train_set_conformity_scores.append(row)
        
        extracted_data = []


        for row in train_set_conformity_scores:
            conformity_score, sample = row[0], row[1]
            extracted_data.append([conformity_score, sample])
            cs = np.array(extracted_data,dtype=object)

        return cs
 

    def pvalue(self, test_sample, cs_list): 
        sorted_indices = np.argsort(cs_list[:, 0])
        sorted_cs_list = cs_list[sorted_indices] 
        rank = 0        
        for i in range(len(sorted_cs_list)):
            if np.array_equal(sorted_cs_list[i][1],test_sample):
                rank = i + 1
                break
        p_value = rank / len(sorted_cs_list)
        
        return p_value

    def get_true_label(self,test_sample):
        for i in range (len(self.X_test)):
            if np.array_equal(test_sample,self.y_test[i]):
                break
            
        return self.y_test[i]
        
    def avg_false_pval_1sample(self,test_sample): # returns avg false p value for only 1 test sample
        possible_labels = self.pos_labels.copy()
        true_label = self.get_true_label(test_sample)
        possible_labels.remove(true_label)
        false_pval = []     
        for i in range (len(possible_labels)):
            label_sample = np.append(test_sample,possible_labels[i])
            cs_list = self.cs_test_sample(label_sample)
            pval = self.pvalue(label_sample,cs_list)
            false_pval.append(pval)
            
        sum = 0
        for i in range (len(false_pval)):
            sum += false_pval[i]
            
        avg_f_pval = sum/len(false_pval)
        
        return avg_f_pval
    
    
    def avg_false_pval(self): # returns avg false p value for a test set.
        avg_false_pval = []
        false_pval = None    
        for i in range(len(self.X_test)):
            sample  = self.X_test[i]
            false_pval = self.avg_false_pval_1sample(sample)
            avg_false_pval.append(false_pval)
            
        sum = 0
        for i in range (len(avg_false_pval)):
            sum += avg_false_pval[i]
            
        avg_f_pval = sum/len(avg_false_pval)
        
        return avg_f_pval

        
    