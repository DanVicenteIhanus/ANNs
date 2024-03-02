import numpy as np, random
from numpy.linalg import lstsq

# ======================= #
# == Helper functions  == #
# ======================= #

def phi(x, mu, sigma) -> np.ndarray:
    return np.exp(-(x - mu)**2/(2*sigma**2))


# ======================= #
# == RBF Network class == #
# ======================= #
class RbfNetwork:
    def __init__(self, sample_size = None, 
                 dim_basis = 1, basis_func = phi, 
                 lr = 0.01, var = 1,
                 epochs = 100) -> None:
        self.sample_size = sample_size
        self.dim_basis = dim_basis
        self.basis_func = basis_func
        self.basis_centers = np.zeros((dim_basis, 1))
        self.basis_sigma = var
        self.lr = lr
        self.epochs = epochs
        self.mses = []
        self.w = np.zeros((dim_basis, 1))
        self.cl = 0

    def _weight_update(self, x, y) -> None:
        self.w += self.lr*(y - self.predict(x))*self.phi_eval(x)
    
    def _init_weights(self) -> None:
        self.w = np.random.uniform(low=-0.1, high=0.1, size=(self.dim_basis, 1))
    
    def _init_weights_CL(self, X) -> None:
        pass

    def _init_basis_random(self) -> None:
        self.basis_centers = np.random.uniform(low=0, high=2*np.pi, size=(self.dim_basis, 1))
        #self.basis_centers = np.linspace(0, 2*np.pi, self.dim_basis)
        
    def compute_phi_matrix(self, X) -> None:
        phi_matrix = np.zeros((X.shape[0], self.dim_basis))
        for i in range(X.shape[0]):
            for j in range(self.dim_basis):
                phi_matrix[i, j] = self.basis_func(X[i], self.basis_centers[j], self.basis_sigma)
        self.phi_matrix = phi_matrix

    def phi_eval(self, x) -> np.ndarray:
        if np.isscalar(x):
            x = np.array([x])
        phi_data = np.zeros((len(x), self.dim_basis))
        for i in range(len(x)):
            for j in range(self.dim_basis):
                phi_data[i, j] = self.basis_func(x[i], self.basis_centers[j], self.basis_sigma)
        return phi_data

    def predict(self, x) -> np.ndarray:
        phi_data = self.phi_eval(x)
        return np.dot(phi_data, self.w).flatten() 
    
    def fit_online(self, X, Y) -> None:
        self._init_basis_random()
        if self.cl > 0:
            self._init_weights_CL()
        else: 
            self._init_weights()
        for _ in range(self.epochs):
            # shuffle data and patterns
            index_shuffle = list(range(X.shape[0]))
            random.shuffle(index_shuffle)
            X = X[index_shuffle]
            Y = Y[index_shuffle]
            # online learning
            for i in range(X.shape[0]):
                y_pred = self.predict(X[i])
                y = Y[i]
                self.w += self.lr*(y - y_pred)*self.phi_eval(X[i]).T
            Y_pred = self.predict(X)
            self.mses.append(np.mean((y_pred - y)**2))

    def fit_lsq(self, X, y) -> None:
        self._init_basis_random()
        if self.cl > 0:
            self._init_weights_CL()
        else: 
            self._init_weights()
        self.compute_phi_matrix(X)
        self.w, _, _, _ = lstsq(self.phi_matrix, y, rcond=None)  # Corrected to unpack the tuple
        self.mses.append(np.mean((self.predict(X) - y)**2))