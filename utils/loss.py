"""
x: numpy array of size mxn
"""
import numpy as np
from abc import abstractmethod

EXTREME = 10e5

class BaseLoss:

    @abstractmethod
    def __init__(self):
        self.search_domain = (-1,1)


    @ abstractmethod
    def __call__(self,x):
        # Calculate loss and return
        pass

    def get_search_domain(self):
        return self.search_domain

class Rastrigin(BaseLoss):

    def __init__(self):
        self.search_domain = (-5.12,5.12)

    def __call__(self, x):
        A = 10
        n = x.shape[1]
        the_sum = np.sum(x**2 - A*np.cos(np.pi), axis=1, keepdims=True)
        result = A*n + the_sum 
        return result

class Sphere(BaseLoss):
    
    def __init__(self):
        self.search_domain = (-EXTREME, +EXTREME)
    
    def __call__(self, x):
        result = np.sum(x*2, axis=1, keepdims=True)
        return result


class Ackley(BaseLoss):

    def __init__(self):
        self.search_domain = (-5,5)
    
    def __call__(self, x):
        x1 = x[:,0].reshape(-1,1)
        x2 = x[:,1].reshape(-1,1)
        first_term = -20*np.exp(-0.2*np.sqrt(0.5*(x1**2+x2**2)))
        second_term = -np.exp(0.5*(np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2)))
        third_term = np.exp(1) + 20
        result = first_term + second_term + third_term
        return result


class Rosenbrock(BaseLoss):

    def __init__(self):
        self.search_domain = (-EXTREME, +EXTREME)

    def __call__(self, x):
        root_term = x[:,:-1]
        cascaded_term = x[:,1:]
        full_term = 100*(cascaded_term - (root_term)**2)**2 + (1 - root_term)**2
        result = np.sum(full_term, axis = 1, keepdims=True)
        return result
    

        
        
        
