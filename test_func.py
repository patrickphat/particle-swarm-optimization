from utils.problem import Sphere, Rosenbrock, Rastrigin, Ackley
from utils.lab import ParticleSwarmOptimization
import numpy as np


losses_map = {
    "sphere": Sphere(),
    "rosenbrock": Rosenbrock(),
    "rastrigin": Rastrigin(),
    "ackley": Ackley(),
}

if __name__ == "__main__":
    problem = Ackley()
    PSO = ParticleSwarmOptimization(pop_size=10000, n_dims=10, problem=problem)
    n_gens = 40
    for i in range(n_gens):
        # print(PSO.pop_mat)
        print(PSO.step())
