from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
from matplotlib import pyplot as py



class PerceptronLearner(SupervisedLearner):
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
    # update the weights
    weights = 0

    def __init__(self):
        pass

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """

        self.labels = []
        # features is the list of data in pair form.
        features.print()
        # labels is the target vector
        labels.print()

        # CONSTANTS that define the LEARNING RATES
        LEARNING_RATE = 0.1
        # How fast the perceptron should converge
        CONVERGENCE_RATE = 0.001
        # how many epochs should we go before increasing the convergence rate?
        INCREASE_CONVERGENCE = 10
        # number of times that we should see no change before we can stop. 5 is arbitrary
        NUM_ROWS_W_NO_WEIGHT_CHANGE = 5

        # make the weights based on the number of columns
        self.weights = [0] * (features.cols + 1)
        bias = 1

        """
        print("Pattern  ", end='       ')
        print("Bias", end=' ')
        print("Target", end=' ')
        print("Weight Vector", end='   ')
        print("Net ", end=' ')
        print("Output ", end=' ')
        print("Change in Weight")

        print("-----------------------------------------------------------------------")
        """
        # I don't know why but when I append the bias to the array everytime it causes errors
        round_one = True
        epoch_count = 0
        # This is the number of changes that we have had so far
        change_count = 0
        converge = 0.0

        # check the number of times that we have not seen a change in weights.
        while change_count < NUM_ROWS_W_NO_WEIGHT_CHANGE:
            # features.shuffle(buddy=labels)
            # go through all the features
            for i in range(features.rows):

                # the pattern or inputs are derived from the features
                pattern = features.row(i)

                # only append the bias the first round
                if round_one:
                    pattern.append(bias)

                # the target is given to us and describes the expected output for each row of the features vector
                target = labels.row(i)[0]

                # do element wise mulitplication of the weight and pattern arrays
                weighted_pattern = np.multiply(self.weights, pattern)
                # sum the result
                net = sum(weighted_pattern)
                # determine output based on summed array
                output = 1 if net > 0 else 0

                # determine the new weights based on the learning rate, target and output
                weight = LEARNING_RATE * (target - output)
                # TODO try and understand what the weight vector really is
                new_weights = np.multiply(pattern, weight)

                # print('{pattern_}  |  {bias_}  |  {target_}  |  {weight_v}  |  {net_}  |  {output_}  |  {change_w}'
                #      .format(pattern_=pattern, bias_=bias, target_=target, weight_v=self.weights, net_=net,
                #              output_=output,
                #             change_w=new_weights))
                # check to see if the change of the weights is less than
                if (np.abs(np.sum(new_weights))) < converge:
                    change_count += 1
                else:
                    change_count = 0

                # take a look at what has changed or not
                self.weights = np.add(new_weights, self.weights)

            round_one = False
            epoch_count += 1
            if epoch_count % INCREASE_CONVERGENCE == 0:
                converge += CONVERGENCE_RATE

            # print("-----------------------------------------------------------------------")

            """
            Determine the output if net is 1 then output is 0
            Net is the matrix multiplied version of pattern and weight vector
            """
        print(epoch_count)

        # graph the algorithm
        # w1*x + w2*y = b graph that as well as the points
        # py.figure()
        # print("Features: ", features.data)
       # for i in range(features.rows):
       #      if labels.row(i)[0] == 1:
       #          py.scatter(features.row(i)[0], features.row(i)[1], c='r')
       #      else:
       #          py.scatter(features.row(i)[0], features.row(i)[1], c='b')
       #
       #  py.grid(color='grey')
       #  py.ylabel('Feature 1')
       #  py.xlabel('Feature 2')
       #
       #
       #  py.title('Perceptron Inputs')
       #  x_range = features.col(0)
       #  # y = (-x*w1 + b)/w2
       #  # print("Weights {}", self.weights)
       #  # print("weights", self.weights[0], self.weights[1], self.weights[2])
       #  m = -1*(self.weights[0]/self.weights[1])
       #  #   x_range = np.arange(-1, 1, 0.1)
       #  # yy = weight_value*x_range + self.weights[2]/self.weights[1]
       #  py.axhline(y=0, c='k')
       #  py.axvline(x=0, c='k')
       #  x = np.linspace(-1, 1)
       #  b = self.weights[2]/self.weights[1]
       #  py.plot(x, m * x + b, linewidth=1)
       #
       #  py.show()



    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """

        # print('Weights: {weights_}'.format(weights_=self.weights))
        # print("Features: {features_} ".format(features_=features))
        if(len(features) != len(self.weights)):
            features.append(1)

        # take a look at what has changed or not
        net = sum(np.multiply(features, self.weights))

        output = 1 if net > 0 else 0
        labels.append(output)
        # print('Labels: {labels_}'.format(labels_=labels))
