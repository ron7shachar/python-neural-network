import numpy as np


class Neuron():
    def __init__(self, id, layer, type):
        self.a = 0
        self.id = id
        self.layer = layer
        self.type = type

    def get_a(self):
        pass

class ActiveNeuron(Neuron):
    def __init__(self, id, layer, type, eta):
        Neuron.__init__(self, id, layer, type)
        self.eta = eta

        self.delta = 0

        self.input_weights = {}
        self.bias = np.random.normal()

        self.mini_batch = 0
        self.delta_input_weights = {}
        self.delta_input_biases = 0

    def get_a(self):
        return self.a

    def feedforward(self):
        """Return the output of the network if ``a`` is input."""
        z = sum(w * neuron.get_a() for neuron, w in self.input_weights.items()) + self.bias
        self.a = sigmoid(z)
        self.sigmoid_prime = sigmoid_prime(z)

    def backward(self):
        self.compute_delta()
        self.delta_input_biases += self.delta
        self.mini_batch += 1
        for neuron in self.input_weights.keys():
            self.delta_input_weights[neuron] += self.delta * neuron.get_a()

    def update_weights_and_bias(self):
        self.bias -= self.delta_input_biases * self.eta / self.mini_batch
        self.delta_input_biases = 0
        for neuron in self.input_weights.keys():
            self.input_weights[neuron] -= self.delta_input_weights[neuron] * self.eta / self.mini_batch
            self.delta_input_weights[neuron] = 0
        self.mini_batch = 0
    def compute_delta(self): pass


class InputNeuron(Neuron):
    def __init__(self, id, layer, type, input):
        Neuron.__init__(self, id, layer, type)
        self.input = input

    def get_a(self):
        return self.input.input[self.id]


class OutputNeuron(ActiveNeuron):
    def __init__(self, id, layer, type, eta):
        ActiveNeuron.__init__(self, id, layer, type, eta)
        self.y = 0

    def compute_delta(self):
        self.delta = (self.a - self.y) * self.sigmoid_prime


class HiddenNeuron(ActiveNeuron):
    def __init__(self, id, layer, type, eta):
        ActiveNeuron.__init__(self, id, layer,type, eta)
        self.outputs = []

    def compute_delta(self):
        self.delta = sum(neuron.input_weights[self] * neuron.delta for neuron in self.outputs) * self.sigmoid_prime



#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
