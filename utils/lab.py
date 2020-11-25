import numpy as np

def initialize_population(pop_size, n_bits, low, high):
    """Initialize population as an numpy array

    Args:
        pop_size (int): Population size
        n_bits ([type]): Number of bits 
        low ([type]): Lower bound of generated number
        high ([type]): Upper bound of generated number

    Returns:
        np.array: An array of sized (pop_size, n_bits)
    """

    population = np.random.uniform(low = low, high = high, size = (pop_size, n_bits))
    return population

class ParticleSwarmOptimization:
    def __init__(self, pop_size, n_bits, low, high, loss):
        self.pop_mat = initialize_population(pop_size = pop_size, n_bits = n_bits, low=low, high)
        self.loss = loss

        # Calculate
        self.inertia = 
        self.cognitive =  
        self.social =

        self.epoch = 0

    def step(self):
        # Calculate

    def get_cognitive(self):
    
    def get_social(self):
        
    def get_inertia(self):