import numpy as np


def initialize_population(pop_size, n_dims, low, high):
    """Initialize population as an numpy array

    Args:
        pop_size (int): Population size
        problem_size ([type]): Number of bits 
        low ([type]): Lower bound of generated number
        high ([type]): Upper bound of generated number

    Returns:
        np.array: An array of sized (pop_size, problem_size)
    """

    population = np.random.uniform(low=low, high=high, size=(pop_size, n_dims))
    return population
