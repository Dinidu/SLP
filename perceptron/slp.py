import numpy as np
from numpy import genfromtxt

class Perceptron(object):

    def __init__(self,numberOfInputFeatures):
        """
        Initialises the weights with random values
        """
        self.wR = np.random.random_sample(numberOfInputFeatures+1)
        self.lr = 0.001
        self.bias = float(1)

    def update_weight(self, inputs, error):
        """
        Adjusts the weights in self.wR
        @param inputs
        @param error
        """
        for x in range(len(inputs)):
            # Adjust the input weights
            self.wR[x] = self.wR[x] + (self.lr * inputs[x] * error)

        # Adjust the bias weight (the last weight in self.w)
        self.wR[-1] = self.wR[-1] + (self.lr * error)

    def weighted_output(self,inputs):
        """
        @param inputs one set of data
        @returns the the sum of inputs multiplied by their weights
        """
        value = 0
        for x in range(len(inputs)):
            # Add the value of the inputs
            value += inputs[x] * self.wR[x]

        # Add the value of bias
        value += self.bias * self.wR[-1]

        return value
    def result(self, inputs):
        """
        @param inputs one set of data
        @returns the the sum of inputs multiplied by their weights
        """
        value = self.weighted_output(inputs)
        # Put value into the SIGMOID equation
        return float(1/(1+np.e ** -value))

    # Utility functions ::
    def sigmoid(x):
        return 1 / (1 + np.e ** -x)

    def recall(self, inputs):
        res = self.result(inputs)
        print("result ",res)
        if res > 0.5:
            return 1
        elif res <= 0.5:
            return 0
        else:
            return 'FAIL'








