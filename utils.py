import numpy as np

## == FUNCTIONS == ##
def phi(x):
        return 2/(1 + np.exp(-x)) - 1

def dphi(x):
    return (1 - phi(x)**2)/2

def sigm(x):
    return 1/(1 + np.exp(-x))

def dsigm(x):
    return sigm(x)*(1 - sigm(x))

def missclassification_error(y_pred, y_true): 
    return np.mean(y_pred != y_true)

def mse(y_pred, y_true): 
    return np.mean((y_true - y_pred)**2)