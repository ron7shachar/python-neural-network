print('######################################  classify_digits ###################################### ')

import mnist_loader
from classify_digits.network import Network
from distributed_neural_network.network1 import Network1
from distributed_levels_neural_network.network2 import  Network2
from unleveled_distributed_neural_network.network3 import  Network3
from genetic_algorithem.genetic import NN_cricher, cricher_maker,GENETIC_ALGORITHM

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()



# net = Network([784, 10, 10])
# net.SGD(training_data, 5, 10, 3.0, test_data=test_data[:1000])

# net2 = Network1([784,10])
# net2.SGD(training_data[:10000], 3, 10, 3.0, test_data=test_data[:1000])

# net3 =  Network2([784,30,10])
#  net3.SGD(training_data, 3, 10, 3.0, test_data=test_data)
#
# net4 = Network3(784,10)
# print(len(training_data))
# net4.SGD(training_data[:10000], 3, 10, 3.0, test_data=test_data[:1000])

problem = {"training":training_data, "validation":validation_data, "test":test_data}
net = Network([784,10,10])
population = cricher_maker(15,net,0.2,0.2,
                  [1000,10000],3,[0.5,5],[1,30])

experiment = GENETIC_ALGORITHM(problem,population,150,0.1,)