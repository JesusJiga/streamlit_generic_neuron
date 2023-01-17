import numpy as np

class Neuron:
    """Esta clase tiene las caracteristicas y el comportamiento de una neurona"""

    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"

    def __init__(self, weights, bias, func):
        self.weights = weights
        self.bias = bias
        self.func = func

    def run(self, input_data):
        y = np.dot(input_data, self.weights) + self.bias
        y = self.__applyActivationFunction(y)
        return y
    
    def __applyActivationFunction(self, value):
        if (self.func == Neuron.RELU):
            return self.__reluFunction(value)
        elif (self.func == Neuron.SIGMOID):
            return self.__sigmoidFunction(value)
        elif (self.func == Neuron.TANH):
            return self.__tanhFunction(value)
        else:
            raise Exception("No existe la función de activación")

    def changeBias(self, bias):
        self.bias = bias
    
    def changeWeights(self, weights):
        self.weights = weights
    
    def changeFA(self, func):
        self.func = func

    @staticmethod
    def __reluFunction(value):
        return max(0.0, value)

    @staticmethod
    def __sigmoidFunction(value):
        return 1/(1 + np.exp(-value))
    
    @staticmethod
    def __tanhFunction(value):
        return np.tanh(value)