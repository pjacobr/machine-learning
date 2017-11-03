from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import math as mat
from matplotlib import pyplot as py

class Node():
    def __init__(self, column_split):
        self.column_split = column_split
        self.children = []

LOG_BASE = 2.0

class DecisionTreeLearner(SupervisedLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """

    """
    Before we can call train, we need to separate the data into feature matrix and label matrix
    """


    labels = []


    def __init__(self):
        pass


    #calculate the entropy of the column that we are using to compare
    def calc_entropy_col(self, data_col, labels, uniq_data_count, uniq_data):
        """
        :type data_col: list
        :type occurences: list
        """
        #print("START")
        #1 go through number of values in the data
        label_count_arrays = []
        #for each value of
        uniq_data_split = []
        #get the indeces of each data value in the arrays
        for data_val in uniq_data:
            data_array = np.where(data_col == data_val)
            uniq_data_split.append(data_array)
            # print(data_array)
            # label_val_count = len(label_data)

        entropy_val = 0

        #
        for array in uniq_data_split:
            label_data = []
            # print("INDECES: ")
            size_array = 0
            for index in np.nditer(array):
                size_array += 1
                label_data.append(labels[index])
            uniq_vals = np.unique(label_data)
            # print("label data: ", label_data)
            label_count = []
            #Add the unique values
            for val in uniq_vals:
                label_count.append(label_data.count(val))
            DATA_VAL_RATIO = size_array/len(data_col)
            # print(len(data_col))
            #ADD the log of the RATIO
            # print("Entropy Change: ")
            # print("Label Count: ", label_count)
            # print("Array: ", array)
            for count in label_count:
                K = count/size_array
                # print("Count: ", count)
                # print("Len Array: ", size_array)
                # print()
                # print("K:", K)
                log_val = mat.log(K, 2.0)
                # print("Log Val: ", log_val)
                entr_temp_val = DATA_VAL_RATIO*(-1*K*mat.log(K,LOG_BASE))
                entropy_val += entr_temp_val
                # print("Ent Val: ", entr_temp_val)
            # print()
        # print(entropy_val)
        return entropy_val
            # print("label_count: ", label_count)






    def calc_unique_vals(self, data_col):
        """
        :type data_col: list
        """
        num_unique = np.unique(data_col)
        counts = []
        # Given the unique values in the array find the number of occurences of each
        for i in num_unique:
            counts.append(data_col.count(i))
        return counts, num_unique

    def calc_entropy(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        output_entropy = 0
        data_entropy = 0
        entropy_vals = []
        for feature_col in range(features.cols):
            uniq_val_occurs, uniq_vals = self.calc_unique_vals(features.col(feature_col))

            # print("Number of Occurences: ", uniq_val_occurs)
            # print("Unique Data Values: ", uniq_vals)
            entropy_val = self.calc_entropy_col(features.col(feature_col), labels.col(0), uniq_val_occurs, uniq_vals)
            entropy_vals.append(entropy_val)

        max_entropy_index = entropy_vals.index(min(entropy_vals))
        if features.cols() > 0:
            #split the data and pass it down the next function
            calc_entropy(Matrix())
            # print()
        return

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """

        self.labels = []
        for i in range(features.rows):
            print(features.row(i), labels.row(i))
        print(self.calc_entropy(features, labels))



    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
