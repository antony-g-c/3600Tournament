import numpy as np

class RatBelief:

    def __init__(self, transition_matrix):
        # The transition matrix T(i, j): probability of moving from i to j
        self.T = np.array(transition_matrix)
        self.belief = np.zeros(64)
        self.coords = np.array([[(i % 8), (i // 8)] for i in range(64)])
        self.NOISE_TABLE = np.array([[0.7, 0.15, 0.15],
                            [0.1, 0.8, 0.1],
                            [0.1, 0.1, 0.8],
                            [0.5, 0.3, 0.2]])

        initial_state = np.zeros(64)
        initial_state[0] = 1.0 
        
        T_1000 = np.linalg.matrix_power(self.T.T, 1000)
        self.belief = T_1000 @ initial_state
            
    def predict(self):
        """Propagate the belief distribution through the transition model."""
        self.belief = self.T.T @ self.belief 
    
    def get_floor_types_array(self, board):
        """
        Extracts the floor types from the board into a 64-element numpy array
        """
        floor_types = np.zeros(64, dtype=np.int32)
        for i in range(64):
            floor_types[i] = board.get_cell((i % 8, i // 8)).value
            
        return floor_types
    
    def update_vectorized(self, floor_types_array, noise, estimated_dist, worker_pos):

        # 1. Vectorized Noise Probabilities
        # We index the NOISE_TABLE array using the floor_types array and the specific noise heard
        P_noise = self.NOISE_TABLE[floor_types_array, noise.value]
        
        # 2. Vectorized Distance Calculations
        # Calculate manhattan distance for all 64 squares simultaneously
        dists = np.abs(self.coords[:, 0] - worker_pos[0]) + np.abs(self.coords[:, 1] - worker_pos[1])
        diffs = estimated_dist - dists
        
        # 3. Vectorized Distance Probabilities
        P_dist = np.zeros(64)
        P_dist[diffs == -1] = 0.12
        P_dist[diffs == 0] = 0.70
        P_dist[diffs == 1] = 0.12
        P_dist[diffs == 2] = 0.06
        
        # Apply the zero-bound edge case simultaneously
        zero_mask = (dists == 0) & (estimated_dist == 0)
        P_dist[zero_mask] = 0.82
        
        # 4. Update and Normalize
        likelihoods = P_noise * P_dist
        self.belief *= likelihoods

        total_belief = np.sum(self.belief)
        if total_belief > 0:
            self.belief /= total_belief

        # 5. Uniform-prior regularization.
        # Iteration-5 tournament showed 0/19 empirical search hit rate
        # against Albert/Carrie despite a 3.0 EV threshold (nominal
        # p_max >= 0.83). The posterior is systematically over-concentrated
        # relative to the true rat distribution — plausibly because the
        # noise + distance observation model's independence assumptions
        # don't hold, or because the transition matrix the harness uses
        # deviates from the one passed in. We mix a small uniform prior
        # into the posterior to de-concentrate. This cannot hurt our
        # local AlbertLite benchmark (we already almost never search there)
        # but should block over-confident tournament searches.
        _UNIFORM_MIX = 0.08     # 8% uniform, 92% posterior
        self.belief = (1.0 - _UNIFORM_MIX) * self.belief + _UNIFORM_MIX / 64.0
    
    def zero_out(self, loc):
        """Zeroes out the probability of a specific cell if we know the rat isn't there."""
        index = loc[1] * 8 + loc[0]
        self.belief[index] = 0.0
        
        # Renormalize the array so it sums to 1.0 again
        total_belief = np.sum(self.belief)
        if total_belief > 0:
            self.belief /= total_belief
    
    def get_best_guess(self):
        """Returns the best cell index to guess and its expected value."""
        best_index = np.argmax(self.belief)
        p = self.belief[best_index]
        ev = (6 * p) - 2
        
        # Convert index back to (x,y)
        x, y = best_index % 8, best_index // 8
        return (x, y), ev