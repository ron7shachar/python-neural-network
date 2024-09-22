from activation_function import ActivationFunction
class Neuron():
    def __init__(self,name,in_weight = {},out_weight = {},bias = 0, activation_function = ActivationFunction(name="") ,alfa = 0.1):
        self.a = 0
        self.z = 0
        self.alfa = alfa
        self.name = name
        self.in_weight = in_weight
        self.out_weight = out_weight
        self.bias = bias
        self.activation_function = activation_function.o_function
        self.d_activation_function = activation_function.d_function

    def forward(self):
        self.z= sum(neuron.value*weight for neuron , weight in self.in_weight.items())
        self.a = self.activation_function(self.z)

    def backward(self):
        for neuron , weight in self.out_weight.items():
            d_zw = self.a
            d_az = self.d_activation_function(neuron.z)
            d_ca





