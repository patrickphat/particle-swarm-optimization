import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, pop_size, problem_size, low, high, loss):
        # Swarm attributes
        self.pop_size = pop_size
        self.problem_size = problem_size
        self.low = low
        self.high = high
        
        # Population and loss used for training
        self.pop_mat = initialize_population(pop_size = pop_size, problem_size = problem_size, low=low, high)
        self.loss = loss

        # Initialize PSO components
        self.current_inertias = np.zeros(size=(pop_size, problem_size)) # First inertias is zeros 
        self.best_positions =  self.pop_mat.copy() # Best position is also the place where it inited
        self.best_position_scores = self.loss(self.best_positions) # Get the best positions score
        self.current_social = np.get_current_social()

        # Weights
        # Inertia weight
        self.w = 0.7298
        # Acceleration constants c
        self.c = 1.49618
        # Number of generations
        self.num_gens = 0

    def step(self):
        # Calculate 
        inertia = self.current_inertia

        # r1 =
        step_direction = self.w*inertia + self.c1*

        self.num_gens += 1 


    def get_new_cognitive(self):
        return
    
    def get_new_social(self):
        """[summary]

        Returns:
            np.array: pop_size x problem_size matrix including social vectors for each 
        """
        padded_population = self._get_padded_population()
        loss = self.loss(padded_population)
        new_social_mat = []

        for i in range(self.pop_size):
            three_cont_indi = loss[i:i+3,:]
            max_idx = np.array(loss)
            best_indi = padded_population[i+max_idx,:]
            new_social_mat.append(best_indi)

        return np.array(new_social_mat)
            



        
    def get_new_inertia(self):


    def _get_padded_population(self):
        first_individual = self.pop_mat[0:1,:]
        last_individual = self.pop_mat[-1:,:]
        padded_population = np.vstack((first_individual,self.pop_mat, last_individual))
        return padded_population
        
        