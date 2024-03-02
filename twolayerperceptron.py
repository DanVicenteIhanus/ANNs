import numpy as np
from utils import *

class TwoLayerPerceptron: 
    def __init__(self, eta: float = 0.01, init_mean: float = 0, alpha: float = 0.9, 
                 init_var: float = 0.1, epochs: float = 100, activation: float = phi, 
                  derivative = dphi, num_nodes: int = 100, beta: float = 0.5):

        self.eta = eta
        self.alpha = alpha
        self.epochs = epochs
        self.mses = []
        self.errors = []
        self.activation = activation
        self.derivative = derivative
        self.init_mean = init_mean
        self.init_var = init_var
        self.z1 = None
        self.z2 = None
        self.a1 = None
        self.a2 = None
        self.W1 = None
        self.W2 = None
        self.theta = None
        self.psi = None
        self.num_nodes = num_nodes
        self.beta = beta

    def __init_weights(self, input_dim) -> None:
        self.W1 = np.random.normal(loc = self.init_mean, scale = np.sqrt(self.init_var), size = (self.num_nodes, input_dim + 1))
        self.W2 = np.random.normal(loc = self.init_mean, scale = np.sqrt(self.init_var), size = (1, self.num_nodes + 1))
        self.theta = np.zeros_like(self.W1)  # Momentum for W1
        self.psi = np.zeros_like(self.W2)    # Momentum for W2

    def __forward_prop(self, X) -> np.ndarray:
        self.z1 = np.dot(self.W1, X) 
        self.a1 = np.concatenate([self.activation(self.z1), np.ones((1, X.shape[1]))]) # " matrix H "
        self.z2 = np.dot(self.W2, self.a1)  
        self.a2 = self.activation(self.z2)                                             # " matrix O "
        return self.a2

    def __back_prop(self, X, Y) -> None:

        # error prop
        Y = Y.reshape(1, -1)
        delta_o = (self.a2 - Y) * self.derivative(self.z2)
        delta_h = np.dot(self.W2.T, delta_o)[:-1, :] * self.derivative(np.vstack([self.z1, np.zeros((1, X.shape[1]))]))[:-1, :]
        
        # compute the momenta
        self.theta = self.alpha * self.theta - (1 - self.alpha) * np.dot(delta_h, X.T)
        self.psi   = self.alpha * self.psi   - (1 - self.alpha) * np.dot(delta_o, self.a1.T)

    def __weight_update(self) -> None:
        self.W1 += self.eta * self.theta
        self.W2 += self.eta * self.psi
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
       self.__init_weights(X.shape[0])
       X = np.vstack([X, np.ones((1, X.shape[1]))])
       for _ in range(self.epochs):
           self.__forward_prop(X)
           self.__back_prop(X, Y)
           self.__weight_update()
           self.mses.append(mse(self.a2, Y))
           self.errors.append(missclassification_error(self.a2, Y))
     
    def predict(self, X) -> np.ndarray:  
        return self.__forward_prop(np.vstack([X, np.ones((1, X.shape[1]))]))