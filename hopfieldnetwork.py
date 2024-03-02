import numpy as np

class HopfieldNet:
    def __init__(self, epochs=10, bias = 0.0, rho = 0.0):
        self.W = None  # Weight matrix
        self.epochs = epochs
        self.energies = []
        self.attractors = None
        self.bias = bias
        self.rho = rho

    def fit(self, X) -> None:
        '''Train the Hopfield network on patterns X (rows).'''
        self.W = X.T @ X
        np.fill_diagonal(self.W, 0)
    
    def fit_sparse(self, X) -> None:
        self.W = (X - self.rho*np.ones_like(X)).T @ (X - self.rho*np.ones_like(X))

    def make_weights_symmetric(self) -> None:
        self.W = 0.5*(self.W + self.W.T)
    
    def synchronous_recall(self, X) -> None:
        '''Performs synchronous updates on the network for given patterns X (rows).'''
        for _ in range(self.epochs):
            X = np.sign(self.W @ X.T).T
            self.energies.append(self.compute_energy(X))

    def asynchronous_recall_single_update(self, X):
        '''Performs an asynchronous update on a single randomly selected neuron in the network for the given pattern X.'''
        if X.ndim == 1:
            X = X.reshape(1, -1)
        for _ in range(X.shape[1]):
            neuron_index = np.random.randint(0, X.shape[1])
            X[0, neuron_index] = np.sign(np.dot(self.W[neuron_index], X.flatten()))
        return X.flatten(), neuron_index

    def asynchronous_recall_with_bias(self, X) -> np.ndarray:
        '''Performs an asynchronous update on the network with bias applied as a threshold.'''
        if X.ndim == 1:
            X = X.reshape(-1)
        for _ in range(len(X)):  # Iterate over neurons
            neuron_index = np.random.randint(0, len(X))
            # Update using bias as a threshold
            net_input = np.dot(self.W[neuron_index], X) - self.bias
            X[neuron_index] = 0.5 + 0.5 * np.sign(net_input)
        return X.flatten()

    def find_num_attractors(self) -> int:
        return len(self.attractors)
    
    def find_attractors(self, patterns) -> None:
        '''Finds and counts the number of unique attractors in the network.'''
        attractors = []
        for pattern in patterns:
            for _ in range(100):
                pattern = self.predict(pattern)
            attractors.append(pattern)
        unique_attractors, unique_indices = np.unique(np.array(attractors), axis=0, return_index=True)
        self.attractors = unique_attractors
    
    def compute_energy(self, X) -> float:
        '''Computes the energy of the network for a given state X.'''
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return -np.sum(np.diag(X @ self.W @ X.T))

    def predict(self, X) -> np.ndarray:
        '''Predicts the state of the network for a given input pattern X.'''
        return np.sign(self.W @ X.T).T