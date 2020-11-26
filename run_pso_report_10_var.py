from utils.problem import Sphere, Rosenbrock, Rastrigin, Ackley
from utils.lab import ParticleSwarmOptimization
from utils.misc import rnd

import numpy as np
import logging
from tqdm import tqdm

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
    pop_sizes = [128, 256, 512, 1024, 2048]
    stat_dict_mean = {f"{i}": {} for i in pop_sizes}
    stat_dict_std = {f"{i}": {} for i in pop_sizes}
    # Get problem
    problem = problem_map[CFGs["problem"]]

    for pop_size in pop_sizes:
        for topology in tqdm(["star", "ring"]):
            cognitive_losses = []

            for problem_name in ["rastrigin", "rosenbrock"]:
                problem = problem_map[problem_name]
                for i in range(10):
                    PSO = ParticleSwarmOptimization(
                        pop_size=pop_size,
                        n_dims=10,
                        problem=problem,
                        topology=topology,
                    )
                    result_dict = PSO.run(max_n_gens=None, max_n_evals=10e6,)
                    cognitive_losses.append(result_dict["cognitive_loss_best"])
                    logger.info(f"Result: {result_dict}")

            stat_dict_mean[f"{pop_size}"][f"{topology}"] = rnd(
                np.mean(cognitive_losses)
            )
            stat_dict_std[f"{pop_size}"][f"{topology}"] = rnd(np.std(cognitive_losses))
            logger.info(f"stat dict mean {stat_dict_mean}")
            logger.info(f"stat dict std {stat_dict_std}")

    with open("logs/report_10_var.log", "a") as fp:
        fp.write(f"Mean: {stat_dict_mean}")
        fp.write(f"Std: {stat_dict_std}")
