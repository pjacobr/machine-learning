from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import matplotlib
from matplotlib import pyplot as py
import csv



# =========== Plot the Vowel MSE based on learning rate/ num epochs
# with open('hidden_nodes.csv', 'rb') as hidden_nodes:
#      read_vowels = csv.reader(hidden_nodes)
#      read_vowels = list(read_vowels)
#      read_vowels = np.array(read_vowels)
#      mse_average_test_set = read_vowels[:, 0]
#      total_mse_to_plot = read_vowels[:, 1]
#      mse_average_v_set = read_vowels[:, 2]
#      num_hidden_nodes = read_vowels[:, 3]
#
#      mse_average_test_set = map(float, mse_average_test_set)
#      total_mse_to_plot = map(float, total_mse_to_plot)
#      mse_average_v_set = map(float, mse_average_v_set)
#      num_hidden_nodes = map(int, num_hidden_nodes)
#
#
#      py.plot(num_hidden_nodes, total_mse_to_plot, 'o', label="Training Set MSE")
#      py.plot(num_hidden_nodes, mse_average_test_set, 'x', label="Test Set MSE")
#      py.plot(num_hidden_nodes, mse_average_v_set, '-', label="Validation Set MSE")
#
#      py.ylabel("MSE (Percent)")
#      py.xlabel("Number of Hidden Nodes")
#      py.legend()
#
# with open('vowel_data.csv', 'rb') as csvfile:
#      read_vowels = csv.reader(csvfile)
#      read_vowels = list(read_vowels)
#      read_vowels = np.array(read_vowels)
#      accuracy_v_set = read_vowels[:, 0]
#      accuracy_test_set = read_vowels[:, 1]
#      mse_average_test_set = read_vowels[:, 2]
#      total_mse_to_plot = read_vowels[:, 3]
#      mse_average_v_set = read_vowels[:, 4]
#      learning_rate = read_vowels[:, 5]
#      num_epochs = read_vowels[:, 6]
#      accuracy_v_set = map(float, accuracy_v_set)
#      accuracy_test_set = map(float, accuracy_test_set)
#      mse_average_test_set = map(float, mse_average_test_set)
#      total_mse_to_plot = map(float, total_mse_to_plot)
#      mse_average_v_set = map(float, mse_average_v_set)
#
#
#      learning_rate = map(float, learning_rate)
#      num_epochs = map(int, num_epochs)
#
#      py.figure()
#      py.plot(learning_rate, total_mse_to_plot, 'o', label="Training Set MSE")
#      py.plot(learning_rate, mse_average_test_set, 'x', label="Test Set MSE")
#      py.plot(learning_rate, mse_average_v_set, '-', label="Validation Set MSE")
#      py.plot(learning_rate, accuracy_test_set, '-', label="Test Set Accuracy")
#      py.plot(learning_rate, accuracy_v_set, '-', label="Validation Set Accuracy")
#
#      py.ylabel("MSE/Accuracy (Percent)")
#      py.xlabel("Learning Rate")
#      py.legend()
#
#      py.figure()
#      py.plot(learning_rate, num_epochs, '-')
#      py.ylabel("Number of Epochs")
#      py.xlabel("Learning Rate")
# =========== Plot Number momentum percentage against the MSE average
with open('layers.csv', 'rb') as momentum_terms:
     read_vowels = csv.reader(momentum_terms)
     read_vowels = list(read_vowels)
     read_vowels = np.array(read_vowels)
     mse_average_test_set = read_vowels[:, 0]
     total_mse_to_plot = read_vowels[:, 1]
     mse_average_v_set = read_vowels[:, 2]
     val_momentum = read_vowels[:, 3]

     mse_average_test_set = map(float, mse_average_test_set)
     total_mse_to_plot = map(float, total_mse_to_plot)
     mse_average_v_set = map(float, mse_average_v_set)
     val_momentum = map(int, val_momentum)


     py.plot(val_momentum, total_mse_to_plot, 'o', label="Training Set MSE")
     py.plot(val_momentum, mse_average_test_set, 'x', label="Test Set MSE")
     py.plot(val_momentum, mse_average_v_set, '-', label="Validation Set MSE")

     py.ylabel("MSE (Percent)")
     py.xlabel("Number of layers")
     py.legend()
py.show()

    #  print(accuracy_v_set)
    #  print(accuracy_test_set)
    #  print(mse_average_test_set)
    #  print(total_mse_to_plot)
    #  print(mse_average_v_set)
    #  print(learning_rate)
    #  print(num_epochs)
