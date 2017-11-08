from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import math as mat
from matplotlib import pyplot as py

global_output_value = 0

class Node():
    def __init__(self, column_split = 0, output_node = False, output_val = 0):
        self.column_split = column_split
        self.children = []
        self.output_val = output_val
        self.output_node = output_node
    def node_print(self):
        print("column_split: ", self.column_split)
        print("children: ", self.children)
        print("output_node", self.output_node)
        print("output_val", self.output_val)
    def prune(self, features, labels, root):
        '''
        features: Matrix
        labels: Matrix
        root: Node
        '''
        if len(self.children) > 0:
            pre_check = 0
            post_check = 0
            for child in self.children:
                if child.output_node == True:
                    pre_check = self.predict(features, labels, root)
                    self.output_node = True
                    break
            if self.output_node == True:
                post_check = self.predict(features, labels, root)
                # print("CHECK: ", post_check, "Pre: ", pre_check)
                #no improvement
                if (post_check - pre_check) <= 0:
                    # print("Not a good output")
                    self.output_node = False
            else:
                for child in self.children:
                    child.prune(features, labels, root)

    def predict(self, features, labels, root):
        """
        :type features: [float]
        :type labels: [float]
        """
        # print("START:::")
        node_pointer = root
        # self.print_tree(node_pointer, 0)
        # print()
        labels_cur = []
        for i in range(features.rows):
            val = self.predict_output(node_pointer, features.row(i))
            labels_cur.append(val)

        count = 0
        for i in range(labels.rows):
            if labels_cur[i] != labels.row(i)[0]:
                count+=1
        return float(count)/float(len(labels_cur))

    def predict_output(self, groot, features):
        # print("Features: ", features)
        # groot.node_print()
        if(groot.output_node == True):
            # print("output", groot.output_val)
            return groot.output_val
        elif(len(groot.children) ==  0):
            # print("output",groot.output_val)
            return groot.output_val
        feature_split = []
        for i in range(len(features)):
            if i != groot.column_split:
                feature_split.append(features[i])
        # print("features: ", features[groot.column_split])
        # print(len(groot.children))
        # print(groot.children)
        # print(groot.column_split)
        index = int(features[int(groot.column_split)])
        if index >= len(groot.children):
            index = 0
        # print(index)
        # print(features)
        self.predict_output(groot.children[index], feature_split)

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
    head_node = Node()

    def __init__(self):
        pass

    # split the partitioned data
    def split_data(self, features, labels, split_column):

        uniq_vals, uniq_val_occurs = np.unique(features.col(split_column), return_counts=True)
        split_data_matrices_indices = []

        for uniq_val in uniq_vals:
            indices = np.where(uniq_val == features.col(split_column))
            split_data_matrices_indices.append(indices)

        new_column_matrix = []
        for column in range(features.cols):
            if column != split_column:
                new_column_matrix.append(np.array(features.col(column)))

        new_column_matrix = np.transpose(new_column_matrix)

        split_data_matrix = []
        split_labels_matrix = []

        for array in split_data_matrices_indices:
            split_matrix = []
            split_labels = []
            for index in array:
                if(len(new_column_matrix) > 0):
                    split_matrix.append(new_column_matrix[index])
                    for i in index:
                        split_labels.append(int(labels.row(i)[0]))
            split_data_matrix.append(split_matrix)
            split_labels_matrix.append(split_labels)

        return_matrix = []
        return_labels_matrix = []
        for index in split_labels_matrix:

            if len(array) > 0:
                label_Matrix = Matrix()
                label_Matrix.set_size(len(index), 1)
                for i in range(len(index)):
                    label_Matrix.set(i, 0, index[i])
                return_labels_matrix.append(label_Matrix)

        for array in split_data_matrix:
            if len(array) > 0:
                inner_Matrix = Matrix()
                inner_Matrix.set_size(len(array[0]), len(array[0][0]))

                for i in range(len(array[0])):
                    for j in range(len(array[0][i])):
                        inner_Matrix.set(i, j, array[0][i][j])
                return_matrix.append(inner_Matrix)

        return return_matrix, return_labels_matrix

    # calculate the entropy of the column that we are using to compare
    def calc_entropy_col(self, data_col, labels, uniq_data_count, uniq_data):
        """
        :type data_col: list
        :type occurences: list
        """
        # print("a1")
        # for each value of
        uniq_data_split = []
        # get the indeces of each data value in the arrays
        for data_val in uniq_data:
            # print("DATA, Va:", data_val)
            data_array = np.where(data_col == data_val)
            uniq_data_split.append(data_array)
            # print(data_array)
            # label_val_count = len(label_data)
        # print("a2")
        entropy_val = 0

        # print("Uniq:" ,uniq_data_split)
        for array in uniq_data_split:
            label_data = []
            # print(array)
            size_array = 0
            if np.size(array) > 0:
                # print("!!!", array)
                for index in np.nditer(array):
                    size_array += 1
                    label_data.append(labels[index])
            uniq_vals, label_count = np.unique(label_data, return_counts=True)

            DATA_VAL_RATIO = size_array / len(data_col)

            for count in label_count:
                K = count / size_array
                log_val = mat.log(K, 2.0)
                # print("Log Val: ", log_val)
                entr_temp_val = DATA_VAL_RATIO * (-1 * K * mat.log(K, LOG_BASE))
                entropy_val += entr_temp_val
                # print("Ent Val: ", entr_temp_val)
                # print()
        # print(entropy_val)
        return entropy_val
        # print("label_count: ", label_count)

    def calc_entropy(self, features, labels, inner_node):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        #BASE Case: If we have only one column left, we can check to see we need to split it or not.
        uniq_labels, labels_counts = np.unique(labels.col(0), return_counts=True)
        uniq_vals, uniq_val_occurs = np.unique(features.col(0), return_counts=True)

        most_common_i = 0
        most_common_val = max(labels_counts)
        for i in range(len(labels_counts)):
            if most_common_val == labels_counts[i]:
                most_common_i = i
                most_common_val = uniq_labels[i]

        inner_node.output_val = most_common_val
        sum = 0
        for label_value in labels_counts:
            sum += label_value

        # if float(most_common_val)/float(label_value) > .9:
        #     inner_node.output_node = True
        #     inner_node.output_val = most_common_val
        #     return inner_node


        if features.cols == 1:
            # print("FE")
            if max(uniq_val_occurs) == 1:
                for uniq_val in uniq_labels:
                    child_node = Node(0, True, uniq_val)
                    inner_node.children.append(child_node)
                return inner_node
        # print("UNIQ:", uniq_labels)
        #if there is only one possible output
        if len(uniq_labels) == 1:
            # print("1")
            inner_node.children.append(Node(0, True, uniq_labels[0]))
            most_common_i = 0
            most_common_val = max(labels_counts)
            for i in range(len(labels_counts)):
                if most_common_val == labels_counts[i]:
                    most_common_i = i
                    most_common_val = uniq_labels[i]
            inner_node.output_val = most_common_val
            # print("OUTPUT: ", uniq_labels[0])
            return inner_node
        # print(2)
        entropy_vals = []
        for i in range(features.cols):
            # print(i)
            entropy_val = self.calc_entropy_col(features.col(i), labels.col(0), uniq_val_occurs, uniq_vals)
            entropy_vals.append(entropy_val)

        max_entropy_index = entropy_vals.index(min(entropy_vals))
        # print("SPLIT ON: ", max_entropy_index)
        split_data, split_labels = self.split_data(features, labels, max_entropy_index)
        # print("THIS ONE HERE")
        # for i in range(len(split_data)):
        #     for row in range(split_data[i].rows):
        #         print(split_data[i].row(row), end=' ')
        #         print(split_labels[i].row(row))
        #     print()
        # print("OUT")

        inner_node.column_split = max_entropy_index
        if features.cols > 0:
            # split the data and pass it down the next function
            for i in range(len(split_data)):
                child_node = Node()
                inner_node.children.append(child_node)
                self.calc_entropy(split_data[i], split_labels[i], child_node)


        return inner_node

    def train(self, features, labels):
        features.shuffle(labels)
        """
        :type features: Matrix
        :type labels: Matrix
        """
        self.head_node = Node()
        self.labels = []
        # for i in range(features.rows):
        #     print(features.row(i), labels.row(i))
        self.calc_entropy(features, labels, self.head_node)
        # self.head_node.prune(features, labels, self.head_node)
        # self.print_tree(self.head_node, 0)
        # self.head_node.prune(features, labels, self.head_node)
        # self.print_tree(self.head_node, 0)
        # self.head_node.prune(features, labels, self.head_node)
        # self.print_tree(self.head_node, 0)
        # self.print_tree(self.head_node, 0)

    def print_tree(self, groot, layer):
        layer += 1

        print(groot.column_split, end=' ')
        if(groot.output_node == True):
            print("* ",groot.output_val, end='')
        for child_node in groot.children:
            self.print_tree(child_node, layer)
        print()

    output_val = 0
    def predict_output(self, groot, features):
        # print("Features: ", features)
        # groot.node_print()
        if(groot.output_node == True):
            # print("output", groot.output_val)
            self.output_val = groot.output_val
            return groot.output_val
        elif(len(groot.children) ==  0):
            # print("output",groot.output_val)
            self.output_val = groot.output_val
            return groot.output_val
        feature_split = []
        for i in range(len(features)):
            if i != groot.column_split:
                feature_split.append(features[i])
        # print("features: ", features[groot.column_split])
        # print(len(groot.children))
        # print(groot.children)
        # print(groot.column_split)
        index = int(features[int(groot.column_split)])
        if index >= len(groot.children):
            index = 0
        # print(index)
        # print(features)
        self.predict_output(groot.children[index], feature_split)




    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        self.output_val = 0
        # print("START:::")
        node_pointer = self.head_node
        # self.print_tree(node_pointer, 0)
        # print()
        self.predict_output(node_pointer, features)

        labels.append(self.output_val)
        # print(labels)
