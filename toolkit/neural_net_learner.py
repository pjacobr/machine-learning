from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import copy as adams_copy
import matplotlib
from matplotlib import pyplot as py


def output_net(net):
    return 1 / (1 + np.exp(-net))


def calc_delta(deltas_above, weight_array, output_of_node):
    return sum(np.multiply(deltas_above, weight_array)) * output_of_node * (1 - output_of_node)


def update_line(hl, x, y):
    hl.set_xdata(np.append(hl.get_xdata(), x))
    hl.set_ydata(np.append(hl.get_ydata(), y))
    py.draw()


def calc_mse_average(targets, output):
    targets = np.array(targets)
    output = np.array(output)
    # print(targets)
    # print(output)
    squared_matrix = targets - output
    squared_matrix = np.square(squared_matrix)
    summed_matrix = np.sum(squared_matrix)
    return summed_matrix


class NeuralNetLearner(SupervisedLearner):
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
    BIAS = 1
    size_of_layers = [8]
    hidden_layers = []
    layer_weights = []
    change_in_weights = []
    deltas = []
    momentum_percentage = 0
    # update the weights
    weights = 0
    learning_rate = .1
    percent_vs = .2

    def __init__(self):
        pass

    def check_validation_set(self, vs_set, vs_labels):
        valid = True

        res_array = []
        mse_diff = 0
        mse_val = 0
        size_target_array = np.unique(vs_labels.col(0))
        for row_index in range(vs_set.rows):
            targets = [0] * size_target_array

            # print("COLLLL,", vs_set.cols)
            targets[int(vs_labels.row(row_index)[0])] = 1
            # print(vs_set.row(row_index))
            instance_result, output_nodes = self.predict_val_set(vs_set.row(row_index))
            res_array.append(instance_result)
            if instance_result != vs_labels.row(row_index)[0]:
                valid = False
            # print("", targets)
            # print("Hidden layer: ",res_array )
            mse_diff_val = calc_mse_average(targets, output_nodes)
            mse_diff += mse_diff_val
            # print("MSE diff val ", mse_diff_val)
        if vs_set.rows != 0:
            # print("MSE Diff: ", mse_diff)
            # print("Num instances: ", vs_set.rows)
            mse_val = mse_diff / vs_set.rows
            # print("MSE Val: ", mse_val)
        else:
            mse_val = 0
        res_array
        vs_labels.col(0)


        diff_count = 0
        if vs_labels.rows == len(res_array):
            for i in range(vs_labels.rows):
                if vs_labels.row(i)[0] != res_array[i]:
                    diff_count += 1
        accuracy = 0
        if vs_labels.rows > 0:
            # print("diff count :", diff_count)
            # print("vs_labels", vs_labels.rows)
            accuracy = 1 - float(diff_count) / float(vs_labels.rows)
            # print("Accuracy", accuracy)
            # print("MSE", mse_val)

            # if valid:
            # print("Results are all valid:", res_array)
        return valid, accuracy, mse_val

    def predict_val_set(self, vs_features):
        """
                :type features: [float]
                :type labels: [float]
                """
        final_layers = adams_copy.deepcopy(self.hidden_layers)
        final_weights = adams_copy.deepcopy(self.layer_weights)
        final_layers[0] = vs_features
        # final_layers[0].append(self.BIAS)

        # print("Final:", final_layers)
        # print("Wieghtst:", final_weights)
        # # print(final_layers[0])
        out_nodes = []
        for index in range(len(final_layers) - 1):
            in_nodes = final_layers[index]
            out_nodes = final_layers[index + 1]

            for j in range(len(out_nodes)):
                if index == len(final_weights) - 1 or j < len(out_nodes) - 1:
                    net_j = np.sum(np.multiply(in_nodes, final_weights[index][j]))
                    out_nodes[j] = output_net(net_j)
                else:
                    break
        res_arr = out_nodes
        # print("Final Layer", final_layers[-1][-1])
        res = res_arr.index(max(res_arr))
        # print(res_arr, "res:", res)
        # print(res, end=', ')
        return res, out_nodes

    def train(self, features, labels, test_features2=Matrix(), test_labels2=Matrix()):
        """
        :type features: Matrix
        :type labels: Matrix
        """

        total_mse_to_plot = []
        features.shuffle(labels)
        test_features = adams_copy.deepcopy(test_features2)
        test_labels = adams_copy.deepcopy(test_labels2)
        test_features.shuffle(test_labels)
        # Append the features
        split_index = int(features.rows * self.percent_vs)
        # print("Split index", split_index)
        vs_matrix = Matrix(features, 0, 0, split_index, features.cols)
        test_matrix = Matrix(features, split_index + 1, 0, features.rows - split_index - 1, features.cols)
        features = test_matrix
        for row_index in range(vs_matrix.rows):
            vs_matrix.row(row_index).append(self.BIAS)
        for row_index in range(test_features.rows):
            test_features.row(row_index).append(self.BIAS)
        # print("Features: ")
        # for i in range(features.rows):
        #     print(features.row(i))
        # print("Labels: ")
        # for i in range(labels.rows):
        #     print(labels.row(i))
        vs_labels_matrix = Matrix(labels, 0, 0, split_index, labels.cols)
        test_labels_matrix = Matrix(labels, split_index + 1, 0, labels.rows - split_index - 1, labels.cols)
        labels = test_labels_matrix
        # [self.hidden_layers.append(features.row(0))
        # self.deltas.append(features.row(0))

        # =========== create data structures ===========
        self.deltas.append([0] * features.cols)
        self.hidden_layers.append([0] * features.cols)
        # print("features\n", self.hidden_layers)
        self.hidden_layers[0].append(1)
        self.deltas[0].append(1)
        # print("features\n", self.hidden_layers)
        # build the neural net based on the number of layers
        # ===========Make the hidden layers =============
        num_epochs = 0
        for i in range(len(self.size_of_layers)):
            layer = []
            layer2 = []
            # number of nodes in that layer
            for j in range(0, self.size_of_layers[i]):
                layer.append(0)
                layer2.append(0)
            # add the BIAS for the input layers
            layer.append(self.BIAS)
            layer2.append(self.BIAS)
            # put all the layers in an array
            self.hidden_layers.append(layer)
            # print("hidden LAYER:\n", self.hidden_layers[0])
            self.deltas.append(layer2)
        # calculate the number of output nodes we need based on the number of
        outputs_size = len(np.unique(labels.col(0)))
        # create the output layer
        self.hidden_layers.append([0] * outputs_size)
        self.deltas.append([0] * outputs_size)

        # print("DELTAS\n", self.deltas)
        # print(self.hidden_layers)
        # self.layer_weights is an array that holds the connections of weights between each of the layers themselves.
        # self.layer_weights = []
        # make sure to set the layers to 1 layer with 2 nodes
        # self.layer_weights = [np.array([[.1, .1, -.1, -.2, .3], [.2, .2, -.1, -.1, -.1]]),
        #                      np.array([[.1, .01, -.3], [.3, -.4, .1], [.5, .1, -.1]])]
        # set the weight arrays for each of the
        # for the last layer there is no bias node so we want to go connect all the nodes....
        output_layer_bias = 1
        # print("hidden layers")
        # ============ Build randomly initialized weights ====================
        for i in range(len(self.hidden_layers) - 1):
            if i >= len(self.hidden_layers) - 2:
                output_layer_bias = 0
            mean, distribution, size_layer_one = 0, 0.1, (len(self.hidden_layers[i]) - 1)
            matrix = np.array([np.random.normal(mean, distribution, size_layer_one + 1) for _ in
                               range(len(self.hidden_layers[i + 1]) - output_layer_bias)])
            self.layer_weights.append(matrix)

        # ========================================================
        #
        # print("self.layer_weights")
        # for i in range(0, len(self.layer_weights)):
        #     for j in range(0, len(self.layer_weights[i])):
        #         # self.layer_weights[i][j][0] = j+1
        #         print(self.layer_weights[i][j])
        #     print()
        #
        # print("done")
        # '''

        targets = [0] * outputs_size
        num_instances = features.rows
        mse_diff = -1
        mse_prev = 0
        mse_average_test_set = []
        mse_average_v_set = []
        accuracy_test_set = []
        accuracy_v_set = []
        #

        while np.abs(mse_diff - mse_prev) > 0.0001:
            features.shuffle(buddy=labels)
            # ================ Begin Algorithm ========================= #
            momentum_weights = []
            mse_average = 0

            # go through all the features in the data set
            for i in range(features.rows):
                # check just on one
                # set the input layer to be the current feature layer that we are working with.
                self.hidden_layers[0][0:features.cols] = features.row(i)
                # PRINT out the hidden layers
                # for l in range(len(self.hidden_layers)):
                #     print(self.hidden_layers[l])
                # iterate through the array of weights and update them

                for j in range(len(self.hidden_layers) - 1):
                    # inputs is the lower layer that we are currently looking at
                    inputs = self.hidden_layers[j]
                    # outputs is the upper layer that we are currently looking at
                    output_nodes = self.hidden_layers[j + 1]
                    # iterate through the upper layer and set the output values based on the input and weights
                    for k in range(len(output_nodes)):
                        # print("input/weights")
                        # print(inputs)
                        # print(self.layer_weights[j][k])
                        # change the layers on the
                        if j == len(self.layer_weights) - 1 or k < (len(output_nodes) - 1):
                            # multiply the inputs by their correspond k < (len(output_nodes) -1) ing weights
                            net = sum(np.multiply(inputs, self.layer_weights[j][k]))
                            output_nodes[k] = output_net(net)
                            # print(output_nodes)
                        else:
                            break

                # ===== Calculate Delta/Weights =====

                # go through the output nodes of our delta and calculate it

                deltas = []
                target = int(labels.row(i)[0])
                targets[target] = 1
                self.change_in_weights = []
                # Go through and fill the output array of deltas
                m_change_weights = []
                for output_delta in range(len(self.hidden_layers[-1])):
                    output_of_node = self.hidden_layers[-1][output_delta]

                    delta = (targets[output_delta] - output_of_node) * output_of_node * (1 - output_of_node)

                    # print((target - output_of_node) * output_of_node * (1 - output_of_node))
                    deltas.append(delta)
                    # print(">> delta:", delta, "\n>> target:", targets[output_delta], "\n>> output: ", output_of_node, "\n", )
                    n_change_in_weights = []
                    for index in range(len(self.hidden_layers[-2])):
                        n_change_in_weights.append(self.hidden_layers[-2][index] * self.learning_rate * delta)

                    m_change_weights.append(n_change_in_weights)

                self.change_in_weights.append(np.array(m_change_weights))
                # -------------------------------------------------------------------------------------------------------------------------------------------------------------

                # print("DELTAS\n", deltas)

                # ====== PRINT CHANGE IN WEIGHTS =========
                # print("change in weights")
                # for h in range(len(self.change_in_weights)):
                #     for w in range(len(self.change_in_weights[h])):
                #         for y in range(len(self.change_in_weights[h][w])):
                #             self.change_in_weights[h][y][w] = y*h*w
                #     print(self.change_in_weights[h][w])
                # print(self.change_in_weights)
                # ========================================
                # ====== PRINT CHANGE IN WEIGHTS =========
                # print("layers")
                # for h in range(len(self.hidden_layers)):
                #     print(self.hidden_layers[h])
                # ========================================

                deltas_above = deltas
                deltas = []

                for layer_index in range(len(self.hidden_layers) - 2, 0, -1):

                    n_change_in_weights = []
                    for x in range(len(self.hidden_layers[layer_index]) - 1):
                        weight_array = self.layer_weights[layer_index][:, x]
                        output_of_node = self.hidden_layers[layer_index][x]
                        calc_delta_result = calc_delta(deltas_above, weight_array, output_of_node)
                        deltas.append(calc_delta_result)
                        # print(">> deltas_above, col (", deltas_above, "*", weight_array)
                        # print(">> output: ", output_of_node)
                        # print(">> delta : ", calc_delta_result)

                    for x in range(len(deltas)):
                        weight_change = []
                        for j_index in range(len(self.hidden_layers[layer_index - 1])):
                            change_weight_result = self.hidden_layers[layer_index - 1][j_index] * self.learning_rate * \
                                                   deltas[x]
                            weight_change.append(change_weight_result)
                            # print(">>> o,d = ", self.hidden_layers[layer_index-1][j_index], ",", deltas[x])
                            # print(">> weight change= ", change_weight_result, "\n")
                        n_change_in_weights.append(weight_change)
                    deltas_above = deltas
                    deltas = []

                    n_change_in_weights = np.array(n_change_in_weights)
                    # print(">>> weight changes <<< ")
                    # for row in n_change_in_weights:
                    #     print(row)
                    self.change_in_weights.append(n_change_in_weights)

                index = 0
                # print("change in weights round 1")
                # for row in reversed(self.change_in_weights):
                #     print(row)
                # print("layer weights round 1")
                # for row in self.layer_weights:
                #     print(row)
                # print("change in weights")
                # for row in self.change_in_weights:
                #     print(row)

                # ==================== MOMENTUM TERMS =======================
                if i < 1:
                    momentum_weights = adams_copy.deepcopy(self.change_in_weights)
                for matrix in reversed(self.change_in_weights):
                    for x in range(len(self.layer_weights[index])):
                        self.layer_weights[index][x] = np.add(self.layer_weights[index][x], matrix[x])
                    index += 1

                index = 0

                momentum_weights = np.multiply(self.momentum_percentage, momentum_weights)
                # for row in momentum_weights:
                #     print(row)
                for matrix in reversed(momentum_weights):
                    for x in range(len(self.layer_weights[index])):
                        self.layer_weights[index][x] = np.add(self.layer_weights[index][x], matrix[x])
                    index += 1

                # print("final weights: ", self.layer_weights)

                mse_average += calc_mse_average(targets, self.hidden_layers[-1])
                momentum_weights = adams_copy.deepcopy(self.change_in_weights)
                targets[target] = 0
                # print("hidden layers round 1")
                # for row in self.hidden_layers:
                #     print(row)

                # # loop through the rest of the delta array and form
                # for layer_index in range(len(self.hidden_layers) - 2, 0, -1):
                #     for x in range(len(self.hidden_layers[layer_index]) - 1):
                #
                #
                #         # the weight array might be
                #
                #         # print(np.outer(output_of_node * self.learning_rate, deltas_above))
                #         # print("WEIGHT CHANGE:", change_weight_array)
                #
                #     #  ============ Calculate change in weights =================
                #
                #     # print("DELTAS", deltas)
            mse_prev = mse_diff
            mse_diff = mse_average / num_instances
            total_mse_to_plot.append(mse_diff)
            # print("HERE")
            valid, accuracy, mse_val = self.check_validation_set(test_features, test_labels)
            mse_average_test_set.append(mse_val)
            accuracy_test_set.append(accuracy)
            # print("here")
            valid, accuracy, mse_val = self.check_validation_set(vs_matrix, vs_labels_matrix)
            accuracy_v_set.append(accuracy)
            mse_average_v_set.append(mse_val)
            if valid:
                break
            num_epochs += 1

        # ============== PLOT GRAPH ==========================
        py.plot(total_mse_to_plot, 'o', label="Training Set MSE")
        py.plot(mse_average_test_set, 'x', label="Test Set MSE")
        py.plot(mse_average_v_set, '-', label="Validation Set MSE")
        py.plot(accuracy_test_set, '-', label="Test Set Accuracy")
        py.plot(accuracy_v_set, '-', label="Validation Set Accuracy")
        py.legend()
        py.show(block=True)
        #=====================================================

    # number of outputs is the last layer
    def predict(self, features, labels):
        # print("\n\n\nCALLLLLL *************************************************\n\n\n")
        """
        :type features: [float]
        :type labels: [float]
        """
        final_layers = adams_copy.deepcopy(self.hidden_layers)
        final_weights = adams_copy.deepcopy(self.layer_weights)
        final_layers[0] = adams_copy.deepcopy(features)
        final_layers[0].append(self.BIAS)

        # print("Final:", final_layers)
        # print("Wieghtst:", final_weights)
        # # print(final_layers[0])
        out_nodes = []
        for index in range(len(final_layers) - 1):
            in_nodes = final_layers[index]
            out_nodes = final_layers[index + 1]

            for j in range(len(out_nodes)):
                # print("\n NODES:")
                # print(in_nodes)
                # print()
                # print(out_nodes)
                # print()
                if index == len(final_weights) - 1 or j < len(out_nodes) - 1:
                    # print(final_weights[index][j])
                    net_j = np.sum(np.multiply(in_nodes, final_weights[index][j]))
                    out_nodes[j] = output_net(net_j)
                else:
                    break
        res_arr = out_nodes
        # print("Final Layer", final_layers[-1][-1])
        res = res_arr.index(max(res_arr))
        # print(res_arr, "res:", res)
        # print(res, end=', ')
        labels.append(res)
