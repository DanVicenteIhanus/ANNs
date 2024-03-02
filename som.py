import numpy as np
# ======================= #
# == SOM Network class == #
# ======================= #

class SelfOrganisingMap:
    
    def __init__(self, epochs: int, num_nodes: int, num_neighbors: int,
                  lr: float, h: any, cyclic: bool, grid_dimension = 1):
        self.W = None
        self.epochs = epochs
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        self.lr = lr
        self.h = h
        self.cyclic = cyclic
        self.grid_dimension = grid_dimension

    def _init_weights(self, num_features) -> None:
        if self.grid_dimension == 1:
            self.W = np.random.uniform(0, 1, size=(self.num_nodes, num_features))
        elif self.grid_dimension == 2:
            self.W = np.random.uniform(0, 1, size=(self.num_nodes, self.num_nodes, num_features))

    def get_winner(self, x) -> int:
        '''input: sample, return: index of closest node in the network'''
        distances = np.sum((x.reshape(1, -1) - self.W)**2, axis=1)
        winner = np.argmin(distances)
        return winner
    
    def get_winner_2d(self, x) -> tuple[int, int]:
        min_dist = np.inf
        winner_index = (0, 0)  # Initialize with a default value
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                dist = np.linalg.norm(x - self.W[i, j, :])
                if dist < min_dist:
                    min_dist = dist
                    winner_index = (i, j)
        return winner_index

    def _weight_update(self, x, winner) -> None: 
        self.W[winner, :] += self.lr*(x - self.W[winner, :])
        for i in range(1, self.num_neighbors + 1):
            
            # update weights for all 
            if winner + i < self.num_nodes:
                self.W[winner + i, :] += self.lr*self.h*(x - self.W[winner + i, :])
            if winner - i >= 0:
                self.W[winner - i, :] += self.lr*self.h*(x - self.W[winner - i, :])
            
            # separate cyclic case to handle neighbors for the first/last node
            if self.cyclic:
                right_neighbor = (winner + i) % self.num_nodes
                left_neighbor = (winner - i) % self.num_nodes
                self.W[right_neighbor, :] += self.lr*self.h*(x - self.W[right_neighbor, :])
                self.W[left_neighbor, :] += self.lr*self.h*(x - self.W[left_neighbor, :])
    
    def _weight_update_2d(self, x, winner_index) -> None:
        
        # update winner
        i, j = winner_index
        self.W[i, j, :] += self.lr*self.h*(x - self.W[i, j, :])
        
        # update its neighbors
        for n in range(1, self.num_neighbors + 1):
            # Need to check all possible neighbors (symmetric 2d grid => 8 cases)
            if (i + n < self.num_nodes):
                self.W[i + n, j, :] += self.lr*self.h*(x - self.W[i + n, j, :])
            if (i + n < self.num_nodes) and (j + n < self.num_nodes):
                self.W[i + n, j + n, :] += self.lr*self.h*(x - self.W[i + n, j + n, :])
            if (i + n < self.num_nodes) and j - n >= 0:
                self.W[i + n, j - n, :] += self.lr*self.h*(x - self.W[i + n, j - n, :])
            if (i - n >= 0):
                self.W[i - n, j, :] += self.lr*self.h*(x - self.W[i - n, j, :])
            if (i - n >= 0 and j + n < self.num_nodes):
               self.W[i - n, j + n, :] += self.lr*self.h*(x - self.W[i - n, j + n, :])
            if (i - n >= 0 and j - n >= 0):
                self.W[i - n, j - n, :] += self.lr*self.h*(x - self.W[i - n, j - n, :])
            if (j + n < self.num_nodes):
                self.W[i, j + n, :] += self.lr*self.h*(x - self.W[i, j + n, :])
            if (j - n >= 0):
                self.W[i, j - n, :] += self.lr*self.h*(x - self.W[i, j - n, :])


    def fit(self, X) -> None:
        '''input X: X[i,:] are the samples and X[:,j] are the features ''' 
        num_samples = X.shape[0]
        num_features = X.shape[1]
        self._init_weights(num_features)
        for epoch in range(self.epochs):
            if self.cyclic:
                if epoch % 10 == 0:
                    self.num_neighbors = max(1, self.num_neighbors - 1)
                if epoch % 15 == 0:
                    self.num_neighbors = max(0, self.num_neighbors - 1)
            else:
                self.num_neighbors = max(1, self.num_neighbors - 2*epoch)
            for i in range(num_samples):
                x = X[i, :]
                winner = self.get_winner(x)
                self._weight_update(x, winner)

    def fit_2d(self, X) -> None:
        num_features = X.shape[1]
        num_samples = X.shape[0]
        self._init_weights(num_features)

        for epoch in range(self.epochs): 
            for i in range(num_samples):   
                x = X[i]
                winner_index = self.get_winner_2d(x)
                self._weight_update_2d(x, winner_index)
            if epoch % 10 == 0:
                self.num_neighbors = max(0, self.num_neighbors - 1)

    def predict(self, X) -> np.ndarray:
        if self.grid_dimension == 1:
            winners = [self.get_winner(x) for x in X]
        elif self.grid_dimension == 2:
            winners = [self.get_winner_2d(x) for x in X]
        else:
            raise ValueError('Unsupported grid dimension.')
        return np.array(winners)