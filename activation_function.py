import numpy as np

def _function_by_name(name):
    match name:
        case 'linear':
            return lambda x:x , lambda x:1
        case 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x)), lambda x:(np.exp(-x)/((1+np.exp(-x))**2))
        case _:
            raise ValueError(f'{name} not a registered function')



class ActivationFunction():
    def __init__(self,o_function = lambda x:x ,d_function = lambda x:1 ,name = ""):
        if name == "":
            self.o_function = o_function
            self.d_function = d_function
        else:
            self.o_function, self.d_function = _function_by_name(name)





