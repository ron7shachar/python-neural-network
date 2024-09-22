import random
import numpy as np
from .neuron import *


class Input():
    def __init__(self):
        self.input = []


class Network1(object):

    def __init__(self, sizes, eta=3.0):
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.neurons = {"input": [], "output": [], "hidden": []}
        self.neurons_ = []
        self.input = Input()
        id = 0
        type = "input"
        for layer in range(self.num_layers):
            if layer == 0:
                type = "input"
                Neuron_ = InputNeuron
                for i in range(self.sizes[layer]):
                    neuron = Neuron_(id, layer, type, self.input)
                    self.neurons[type].append(neuron)
                    self.neurons_.append(neuron)
                    id += 1
                continue
            if layer == self.num_layers - 1:
                type = "output"
                Neuron_ = OutputNeuron
            elif self.num_layers > 2:
                type = "hidden"
                Neuron_ = HiddenNeuron
            for i in range(self.sizes[layer]):
                neuron = Neuron_(id, layer,type, eta)
                self.neurons[type].append(neuron)
                self.neurons_.append(neuron)
                id += 1

        for neuron in self.neurons["hidden"]:
            neuron.bias = np.random.normal()
        for neuron in self.neurons["output"]:
            neuron.bias = np.random.normal()
        self.hidden_layer_size = len(self.neurons["hidden"])
        weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        id = 0
        for layer, layer_size in enumerate(self.sizes[1:]):
            id_ = id
            id += self.sizes[layer]
            for neuron in self.neurons_[id:id + layer_size]:
                for id__, weight in enumerate(weights[layer][neuron.id - id]):
                    neuron.input_weights[self.neurons_[id_ + id__]] = float(weight)
                    neuron.delta_input_weights[self.neurons_[id_ + id__]] = 0

                    if self.neurons_[id_ + id__].layer != 0:
                        self.neurons_[id_ + id__].outputs.append(neuron)

    def prediction(self, x):
        prediction = []
        """Return the output of the network if ``a`` is input."""
        self.input.input = x
        for neuron in self.neurons["hidden"]:
            neuron.feedforward()
        max = self.neurons["output"][0].a
        prediction = 0
        for i, neuron in enumerate(self.neurons["output"]):
            neuron.feedforward()
            if neuron.a > max:
                max = neuron.a
                prediction = i
        return prediction

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
                self.update_weights_and_bias()
            print(f"Epoch {j + 1}")
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            # else:
            #     print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        for x, y in mini_batch:
            self.backprop([x_[0] for x_ in x], [y_[0] for y_ in y])
        #
        #     nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        #     nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # self.weights = [w-(eta/len(mini_batch))*nw
        #                 for w, nw in zip(self.weights, nabla_w)]
        # self.biases = [b-(eta/len(mini_batch))*nb
        #                for b, nb in zip(self.biases, nabla_b)]
        #

    def backprop(self, x, y):
        self.input.input = x
        for neuron in self.neurons["hidden"]:
            neuron.feedforward()
        for i, y_ in enumerate(y):
            self.neurons["output"][i].y = y_
            self.neurons["output"][i].feedforward()
        for neuron in self.neurons["output"]:
            neuron.backward()
        for i in range(self.hidden_layer_size):
            self.neurons["hidden"][self.hidden_layer_size - i - 1].backward()

        pass

    def update_weights_and_bias(self):
        for neuron in self.neurons["hidden"]:
            neuron.update_weights_and_bias()
        for neuron in self.neurons["output"]:
            neuron.update_weights_and_bias()

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(self.prediction(x), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
