from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import math as mat
from matplotlib import pyplot as py

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

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        self.labels = []

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
