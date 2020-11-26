from .population import initialize_population
from .problem import BaseProblem
from .misc import rnd

import logging
import numpy as np
import matplotlib.pyplot as plt
import tempfile

from PIL import Image
import imageio as io

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ParticleSwarmOptimization:
    def __init__(self, pop_size:int, n_dims:int, problem:BaseProblem, topology:str = "ring"):
        """Particle Swarm Optimization

        Args:
            pop_size (int): Size of PSO population
            n_dims (int): Number of dimensions
            problem (BaseProblem): Problem function to be solved
            topology (str, optional): Neighborhood topology. Defaults to "ring".
        """
        # Swarm attributes
        self.pop_size = pop_size
        self.n_dims = n_dims
        
        # Neighborhood topology
        self.topology = topology

        # Search domain
        search_domain = problem.get_search_domain()
        self.low = search_domain[0]
        self.high = search_domain[1]

        # Population and loss used for training
        self.pop_mat = initialize_population(
            pop_size=pop_size, n_dims=n_dims, low=self.low, high=self.high
        )
        self.problem = problem

        # Number of evaluation called
        self.n_evals = 0

        # Initialize PSO components
        self.current_inertias = np.zeros(shape=(pop_size, n_dims))

        # First inertias is zeros
        self.current_cognitives = self.pop_mat.copy()

        # Best position loss is also the place where it inited
        self.current_cognitive_losses = self.evaluate(self.current_cognitives)

        # Get the best positions score
        self.current_social = self.update_social()

        # Inertia weight
        self.w = 0.3

        # Acceleration constant c
        self.c = 1.49618

        # Number of generations
        self.n_gens = 0

    def step(self):

        # Get current base inertia
        inertia = self.current_inertias

        # Retrieve and calculate cognitive vector displacement
        cognitive = self.current_cognitives
        cognitive_displacement = self.current_cognitives - self.pop_mat

        # Retrieve and calculate social vector displacement
        social = self.current_social
        social_displacement = self.current_social - self.pop_mat

        # Generate noises for displacement
        r_noise_1 = np.random.uniform(size=(self.pop_size, self.n_dims))
        r_noise_2 = np.random.uniform(size=(self.pop_size, self.n_dims))

        # Calculate and step the population in a new direction
        step_direction = self.w * inertia + self.c * (r_noise_1 * cognitive_displacement + r_noise_2 * social_displacement)
        self.pop_mat += step_direction

        # Update some properties of the population
        self.current_inertias = self.update_inertia(step_direction)
        self.current_cognitives = self.update_cognitive()
        self.current_social = self.update_social()

        self.n_gens += 1

        current_losses = self.evaluate(self.pop_mat)
        cognitive_losses = self.evaluate(self.current_cognitives)
        dist_to_opt = self.problem.distance_to_optimal(self.pop_mat)

        result_dict = {
            "cognitive_loss_best": rnd(np.min(cognitive_losses)),
            "distance_to_opt_best": rnd(np.min(dist_to_opt)),
            "n_evals": rnd(self.n_evals), 
            "n_gens": rnd(self.n_gens)
            }
        return result_dict

    def is_exceed(self, max_n_gens, max_n_evals):
        if max_n_gens:
            is_exceed_max_n_gens = self.n_gens + 1 > max_n_gens
        else:
            is_exceed_max_n_gens = False
        
        if max_n_evals:
            is_exceed_max_n_evals = self.n_evals + self.pop_size > max_n_evals
        else:
            is_exceed_max_n_evals = False

        if is_exceed_max_n_evals or is_exceed_max_n_gens:
            return True

        return False

    def draw_img(self):
        n_samples = 100

        xlist = np.linspace(self.low, self.high, n_samples)
        ylist = np.linspace(self.low, self.high, n_samples)
        X, Y = np.meshgrid(xlist, ylist)

        X_ = X.reshape((n_samples**2,1))
        Y_ = Y.reshape((n_samples**2,1))
        Z = self.problem.evaluate(np.hstack((X_,Y_)))
        Z = Z.reshape((n_samples,n_samples))

        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(X, Y, Z)
        fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title(f'Visualization gen {self.n_gens}')
        ax.scatter(self.pop_mat[:,0],self.pop_mat[:,1], color = "white", s=4)
        ax.set(xlim=(self.low, self.high), ylim=(self.low, self.high))


        temp_img_path = tempfile.NamedTemporaryFile()

        fig.savefig(temp_img_path)
        img = np.array(Image.open(temp_img_path))
        return img

    def run(self, print_log_every:int=None, max_n_gens:int = 50, max_n_evals:int = 10e6, export_gif:str=None):
        if export_gif:
            viz_images = []

        i = 0
        result_dict = {}
        while not self.is_exceed(max_n_gens, max_n_evals):
            if export_gif:
                img = self.draw_img()
                viz_images.append(img)

            i += 1
            result_dict = self.step()
            if print_log_every:
                if i % print_log_every == 0:
                    logger.info(f"::[GEN {self.n_gens}] {result_dict}]")
        
        if export_gif:
            io.mimsave(export_gif, viz_images, duration = 0.5)

        return result_dict

    def update_cognitive(self):
        """
        Returns:
            np.array: pop_size x n_dims matrix including cognitives vectors for each individuals 
        """
        new_pop_losses = self.evaluate(self.pop_mat)
        is_better = (new_pop_losses < self.current_cognitive_losses)[:,0]
        #print(is_better)

        for i, bool_ in enumerate(is_better):
            # If it is better then drop
            if bool_:
                self.current_cognitives[i] = self.pop_mat[i]
                self.current_cognitive_losses[i] = new_pop_losses[i]

        return self.current_cognitives

    def update_social(self):
        """
        Returns:
            np.array: pop_size x n_dims matrix including social vectors for each individuals 
        """
        padded_population = self._get_padded_population()
        loss = self._pad_matrix(self.evaluate(padded_population))

        if self.topology == "ring":
            new_social_mat = []

            for i in range(self.pop_size):
                # Get 3 consecutive individual
                three_cont_loss = loss[i : i + 3, :]

                # Get the fittest individual position
                best_idx = np.argmin(three_cont_loss)

                # Get best individual 
                best_indi = padded_population[i + best_idx, :]

                # Append that individual to social matrx
                new_social_mat.append(best_indi)

            return np.array(new_social_mat)

        elif self.topology == "star":
            # Get the fittest individual position
            best_idx = np.argmin(loss)
            best_indi = padded_population[best_idx]
            return np.tile(best_indi, (self.pop_size,1))

    def update_inertia(self, x):
        self.current_inertias = x.copy()
        return self.current_inertias

    def _get_padded_population(self):
        return self._pad_matrix(self.pop_mat)

    @staticmethod
    def _pad_matrix(x):
        first_entry = x[0:1, :]
        last_entry = x[-1:, :]
        padded_matrix = np.vstack((last_entry, x, first_entry))
        return padded_matrix

    def evaluate(self, pop_mat, skip_count = False):
        loss = self.problem.evaluate(pop_mat)
        num_indi = pop_mat.shape[0]
        
        if not skip_count:
            self.n_evals += num_indi
        return loss
