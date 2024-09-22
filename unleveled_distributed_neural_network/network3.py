import random
import numpy as np
from .neuron import *


class Input():
    def __init__(self):
        self.input = []



class Network3(object):

    def __init__(self, input , output, eta=3.0):

        self.neurons = {}
        self.input = Input()
        self.id = 0
        self.neurons = {"input": [], "output": [], "hidden": []}
        for i in range(input):
            neuron = InputNeuron(self.id, 0,  "input", self.input)
            self.neurons["input"].append(neuron)
            self.id += 1


        for i in range(output):
            neuron = OutputNeuron(self.id, 1 ,"output", eta , 0.2)
            neuron.bias = np.random.normal()
            neuron.weights = {neuron_:Weight(random.normalvariate()) for neuron_ in self.neurons["input"] }
            self.neurons["output"].append(neuron)
            self.id += 1


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
            self.summary()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
                self.update_weights_and_bias()
            print(f"Epoch {j + 1}")
            if test_data:
                print("pre Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
                self.evolve(n_test)
                print("after Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            # else:
            #     print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        for x, y in mini_batch:
            self.backprop([x_[0] for x_ in x], [y_[0] for y_ in y])


    def backprop(self, x, y):
        self.input.input = x
        for neuron in self.neurons["hidden"]:
            neuron.feedforward()
        for i, y_ in enumerate(y):
            self.neurons["output"][i].y = y_
            self.neurons["output"][i].feedforward()
        for neuron in self.neurons["output"]:
            neuron.backward()
        for layer in self.neurons["hidden"]:
            for neuron in layer:
                self.neurons["hidden"].backward()

        pass

    def update_weights_and_bias(self):
        for neuron in self.neurons["hidden"]:
            neuron.update_weights_and_bias()
        for neuron in self.neurons["output"]:
            neuron.update_weights_and_bias()

    def evolve(self,n):
        for neuron in self.neurons["output"]:
            neuron.evolve(n)
        for layer in self.neurons["hidden"]:
            for neuron in layer:
                self.neurons["hidden"].evolve()


    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(self.prediction(x), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def summary(self):

        size = 0
        var = 0
        w = 0

        print("_______________ neurons by layer _______________")
        print(f" input : {len(self.neurons["input"])} neurons")
        for i ,h_l in self.neurons["hidden"]:
            connections = 0
            for neuron in enumerate(h_l):
                connections += len(neuron.weights)
                size += 1
                var += sum(weight.var for weight in neuron.weights.values()) / len(neuron.weights)
                w = sum(abs(weight.value_) for weight in neuron.weights.values()) / len(neuron.weights)
            print(f'number of connections : {connections}')
            print(f" hidden {i} : {len(h_l)} neurons")
        connections = 0
        for neuron in self.neurons["output"]:
            connections += len(neuron.weights)
            size += 1
            var += sum(weight.var for weight in neuron.weights.values()) / len(neuron.weights)
            w = sum(abs(weight.value_) for weight in neuron.weights.values()) / len(neuron.weights)
        print(f'number of connections : {connections}')
        print(f" output : {len(self.neurons["output"])} neurons")

        print("_____________________ summary _____________________")
        print(f" average weight : {w/size}  |  average var : {var/size}  ")