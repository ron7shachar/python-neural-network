import random

import numpy as np
# import scipy.stats as stats

class Weight():
    def __init__(self,value_, delta = 0, var = 0):
        self.value_ = value_
        self.delta = delta
        self.var = var

    def __mul__(self, other):
        return self.value_*other



class Neuron():
    def __init__(self, id, layer, type):
        self.a = 0
        self.id = id
        self.layer = layer
        self.type = type


    def __repr__(self):
        return f"id (self.id): {self.id}, layer {self.layer}, type {self.type})"
    def get_a(self):
        pass

class ActiveNeuron(Neuron):
    def __init__(self, id, layer, type, eta , T):
        Neuron.__init__(self, id, layer, type)
        self.eta = eta
        self.T = T
        self.delta = 0

        self.weights = {}
        self.bias = np.random.normal()
        self.outputs = []

        self.mini_batch = 0
        self.delta_input_biases = 0


    def get_a(self):
        return self.a

    def feedforward(self):
        """Return the output of the network if ``a`` is input."""
        z = sum(w * neuron.get_a() for neuron, w in self.weights.items()) + self.bias
        self.a = sigmoid(z)
        self.sigmoid_prime = sigmoid_prime(z)

    def backward(self):
        self.compute_delta()
        self.delta_input_biases += self.delta
        self.mini_batch += 1
        for neuron in self.weights.keys():
            self.weights[neuron].delta += self.delta * neuron.get_a()

    def update_weights_and_bias(self):
        self.bias -= self.delta_input_biases * self.eta / self.mini_batch
        self.delta_input_biases = 0
        for neuron in self.weights.keys():
            delta = self.weights[neuron].delta * self.eta / self.mini_batch
            self.weights[neuron].var += delta**2
            self.weights[neuron].value_ -= delta
            self.weights[neuron].delta = 0.0
        self.mini_batch = 0

    def compute_delta(self): pass

    def evolve(self,n):
        remove = []
        extend = []
        for neuron , weight in self.weights.items():
            if self.decision_r(weight):
                remove.append(neuron)
            if self.decision_e(weight,n):
                extend.append(neuron)


        for neuron in remove:
            self.weights.pop(neuron)
            if neuron.type != "input":
                neuron.outputs.remove(self)



        self.T_F()



    def decision_r(self, weight):
        return random.random() > np.exp(-abs(self.T/weight.value_))

    def T_F(self):
        self.T = 0

    def decision_e(self, weight, n):
        var = (weight.var / n)**0.5



class InputNeuron(Neuron):
    def __init__(self, id, layer, type, input):
        Neuron.__init__(self, id, layer, type)
        self.input = input

    def get_a(self):
        return self.input.input[self.id]


class OutputNeuron(ActiveNeuron):
    def __init__(self, id, layer, type, eta,T):
        ActiveNeuron.__init__(self, id, layer, type, eta,T)
        self.y = 0

    def compute_delta(self):
        self.delta = (self.a - self.y) * self.sigmoid_prime


class HiddenNeuron(ActiveNeuron):
    def __init__(self, id, layer, type, eta):
        ActiveNeuron.__init__(self, id, layer,type, eta)


    def compute_delta(self):
        self.delta = sum(neuron.input_weights[self] * neuron.delta for neuron in self.outputs) * self.sigmoid_prime



#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
