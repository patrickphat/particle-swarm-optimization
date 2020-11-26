from utils.problem import Sphere, Rosenbrock, Rastrigin, Ackley
from utils.lab import ParticleSwarmOptimization

import numpy as np
import logging

CFGs = {
    "pop_size": 32,
    "n_dims": 2,
    "problem": "ackley",
    "max_n_gens": 50,
    "max_n_evals": None,
    "topology": "star",
    "n_experiments": 1,
    "export_gif": None,
    "log_path": "logs/experiment_2_var.log",
}

BASE_RANDOM_SEED = 17520880
np.random.seed(BASE_RANDOM_SEED)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

problem_map = {
    "sphere": Sphere(),
    "rosenbrock": Rosenbrock(),
    "rastrigin": Rastrigin(),
    "ackley": Ackley(),
}

if __name__ == "__main__":

    with open(CFGs["log_path"], "a") as fp:
        fp.write(f"[Experiment] CFGs: {CFGs}\n")

    # Get problem
    problem = problem_map[CFGs["problem"]]

    for i in range(CFGs["n_experiments"]):
        PSO = ParticleSwarmOptimization(
            pop_size=CFGs["pop_size"], 
            n_dims=CFGs["n_dims"], 
            problem=problem, 
            topology=CFGs["topology"]
            )
        result_dict = PSO.run(
            max_n_gens=CFGs["max_n_gens"],
            max_n_evals=CFGs["max_n_evals"],
            export_gif=CFGs["export_gif"]
            )

        with open(CFGs["log_path"], "a") as fp:
            idx_str = str(i + 1).zfill(2)
            fp.write(f"Run #{idx_str}: {result_dict}\n")

        logger.info(f"Result: {result_dict}")   