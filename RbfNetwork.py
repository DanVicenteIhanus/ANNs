import numpy as np, random
from numpy.linalg import lstsq
from DD2437_Lab2_script import *



# ======================= #
# == Helper functions  == #
# ======================= #

def f_1(x, noise = 0) -> np.ndarray: 
    return np.sin(2*x) if noise == 0 else np.sin(2*x) + np.random.normal(loc=0, scale=noise, size=np.shape(x))

def f_2(x, noise = 0) -> np.ndarray: 
    f_arr = np.zeros_like(x)
    for i, d in enumerate(x):
        f_arr[i] = 1 if np.sin(d) >= 0 else -1
    return f_arr if noise == 0 else f_arr + np.random.normal(loc=0, scale=noise, size=np.shape(x))

def gen_func_data(f, start, stop, step_length, var = 0) -> np.ndarray:
    '''Generate training/test-data given a function f'''
    training_interval = np.arange(start = start, stop = stop, step= step_length)
    test_interval = np.arange(start = start + 0.05, stop = stop, step = step_length)
    return training_interval, f(training_interval, var), test_interval, f(test_interval, var)

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


# ================= #
# == Simulations == #
# ================= #
def plot_functions(ax) -> None:
    ax.plot(x_2_test, f_1_test_pred_online, c='b', lw=1.3, ls = 'dashed', label='Predictions [on-line]')
    ax.plot(x_2_test, f_1_test_pred_batch, c='#bf9000', lw=1.3, ls = 'dashed', label='Predictions [batch lsq]')
    ax.plot(x_2_test, f_1_test, c='r', lw=1.5, alpha=0.7, label='exact')
    ax.legend(loc='upper right')
    ax.set_xlim([0,2*np.pi])
    print(f'MSE with online [epochs = {rbf_network.epochs}] = {round(rbf_network.mses[-1], 4)}')
    print(f'MSE with batch = {round(rbf_network_batch.mses[-1], 4)}')

if __name__ == '__main__':
    start = 0.0
    stop = 2*np.pi
    step_length = 0.1
    dim_basis = 5
    var = 0.1
    epochs = 100
    
    x_1_training, f_1_training, x_1_test, f_1_test = gen_func_data(f_1, start, stop, step_length, var)
    x_2_training, f_2_training, x_2_test, f_2_test = gen_func_data(f_2, start, stop, step_length, var)
    rbf_network = RbfNetwork(f_1_training.shape[0], dim_basis, phi, 0.1, 1.0, epochs)
    rbf_network_batch = RbfNetwork(f_1_training.shape[0], dim_basis, phi, 0.1, 1.0, epochs)

    rbf_network.fit_online(x_1_training, f_1_training)
    f_1_test_pred_online = rbf_network.predict(x_1_test)

    rbf_network_batch.fit_lsq(x_1_training, f_1_training)
    f_1_test_pred_batch = rbf_network_batch.predict(x_1_test)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    plot_functions(ax)