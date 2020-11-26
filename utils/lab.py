import numpy as np
from .population import initialize_population


class ParticleSwarmOptimization:
    def __init__(self, pop_size, n_dims, problem):
        # Swarm attributes
        self.pop_size = pop_size
        self.n_dims = n_dims

        search_domain = problem.get_search_domain()
        self.low = search_domain[0]
        self.high = search_domain[1]

        # Population and loss used for training
        self.pop_mat = initialize_population(
            pop_size=pop_size, n_dims=n_dims, low=self.low, high=self.high
        )
        self.problem = problem

        # Number of evaluation called
        self.n_evaluations = 0

        # Initialize PSO components
        self.current_inertias = np.zeros(shape=(pop_size, n_dims))

        # First inertias is zeros
        self.current_cognitives = self.pop_mat.copy()

        # Best position loss is also the place where it inited
        self.current_cognitive_losses = self.evaluate(self.current_cognitives)

        # Get the best positions score
        self.current_social = self.update_social()
        # import ipdb

        # ipdb.set_trace()

        # Inertia weight
        self.w = 0.3

        # Acceleration constant c
        self.c = 1.49618

        # Number of generations
        self.num_gens = 0

    def step(self):

        # Get current base inertia
        inertia = self.current_inertias

        cognitive = self.current_cognitives
        cognitive_displacement = self.current_cognitives - self.pop_mat

        social = self.current_social
        social_displacement = self.current_social - self.pop_mat

        r_noise_1 = np.random.uniform(size=(self.pop_size, self.n_dims))
        r_noise_2 = np.random.uniform(size=(self.pop_size, self.n_dims))
        step_direction = self.w * inertia + self.c * (
            r_noise_1 * cognitive_displacement + r_noise_2 * social_displacement
        )
        self.pop_mat += step_direction

        # Update
        self.current_inertias = self.update_inertia(step_direction)
        self.current_cognitives = self.update_cognitive()
        self.current_social = self.update_social()

        self.num_gens += 1

        losses = self.evaluate(self.pop_mat)
        # print("LOSSES: ", losses)
        result_dict = {"mean": np.mean(losses), "min": np.min(losses)}
        return result_dict

    def update_cognitive(self):
        new_cognitive_losses = self.evaluate(self.pop_mat)
        is_better = new_cognitive_losses < self.current_cognitive_losses

        for i, bool_ in enumerate(is_better):
            # It is better then drop
            # import ipdb

            # ipdb.set_trace()
            if bool_:
                self.current_cognitives[i] = self.pop_mat[i]
                self.current_cognitive_losses[i] = new_cognitive_losses[i]

        return self.current_cognitives

    def update_social(self):
        """
        Returns:
            np.array: pop_size x n_dims matrix including social vectors for each 
        """
        padded_population = self._get_padded_population()
        loss = self._pad_matrix(self.evaluate(padded_population))

        new_social_mat = []

        for i in range(self.pop_size):
            three_cont_loss = loss[i : i + 3, :]

            max_idx = np.argmax(three_cont_loss)
            try:
                best_indi = padded_population[i + max_idx, :]
            except:
                import ipdb

                ipdb.set_trace()
            new_social_mat.append(best_indi)

        return np.array(new_social_mat)

    def update_inertia(self, x):
        self.current_cognitives = x
        return self.current_cognitives

    def _get_padded_population(self):
        return self._pad_matrix(self.pop_mat)

    @staticmethod
    def _pad_matrix(x):
        first_entry = x[0:1, :]
        last_entry = x[-1:, :]
        padded_matrix = np.vstack((last_entry, x, first_entry))
        return padded_matrix

    def evaluate(self, pop_mat):
        loss = self.problem.evaluate(pop_mat)
        self.n_evaluations += self.pop_size
        return loss
