
from utils.loss import Sphere, Rosenbrock, Rastrigin, Ackley
from utils.population import initialize_population
import numpy as np


losses_map = {
    "sphere": Sphere(),
    "rosenbrock": Rosenbrock(),
    "rastrigin": Rastrigin(),
    "ackley": Ackley(),
}

def 