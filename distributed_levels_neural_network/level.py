import numpy as np


class Lavel():
    def __init__(self,id,type):
        self.id = id
        self.neurons = []
        self.weights = []
        self.biases = []
        self.activations = []
        self.zs = []
        self.type = type

    def feedforward(self, x):
        activation = x
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, x)+b
            activation = sigmoid(z)
            self.activations.append(activation)
            self.zs.append(z)

    def backprop(self,error,pre_activation,post_delta):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = np.zeros(self.biases.shape)
        nabla_w = np.zeros(self.weights.shape)
        # backward pass
        if self.type == 'output':
            # delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
            delta = error * sigmoid_prime(self.zs)
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, pre_activation.transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        else:
            sp = sigmoid_prime(self.zs)
            delta = np.dot(self.weights[-l + 1].transpose(), post_delta) * sp
            nabla_b = delta
            nabla_w = np.dot(delta, pre_activation.transpose())
        return (nabla_b, nabla_w)
#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))