'''
Perceptron with sequential and batch training methods
'''

import numpy as np

class Perceptron: 
    def __init__(self, eta = 1, bias = 0.0,
                  mean = 0.0, var = 0.001, epochs = None,
                    training = 'perceptron', max_iter = 1000):
        self.w = None
        self.iteration = 0
        self.eta = eta
        self.w0 = bias
        self.mean = mean
        self.epochs = epochs
        self.training = training
        self.mse = []
        self.errors = []
        self.max_iter = max_iter
        self.var = var
    
    def fit_batch(self, data):
        # Initialize weights including the bias weight
        W = np.random.normal(loc = self.mean, scale=self.var, size = data.shape[1])
        X = data.drop('y', axis=1).values
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        y = data['y'].values
        for _ in range(self.epochs):
            y_pred = np.dot(X, W)
            errors = y_pred - y
            
            # Compute and store MSE using the continuous outputs
            mse_epoch = np.mean(errors**2)
            self.mse.append(mse_epoch)
            
            # Apply thresholding when calculating misclassification error
            y_class = np.where(y_pred < 0, -1, 1)
            misclassification_epoch = np.mean(y_class != y)
            self.errors.append(misclassification_epoch)
            
            # Update weights using the delta rule
            W -= self.eta * np.dot(X.T, errors)
        
        # Update the bias (first weight) and the rest of the weights
        self.w0 = W[0]
        self.w = W[1:]

    
    def fit_sequential(self, data):
        self.w = np.random.normal(loc=0, scale= self.var, size=len(data.columns) - 1)
        while True:
            mse_total = 0  # Initialize total MSE for this iteration
            missclassification_total = 0  # Initialize total misclassification error for this iteration
            for _, row in data.iterrows():
                xi = np.hstack([1, row.drop('y').values])  # Include bias in the input vector
                y = row['y']
                y_pred = np.dot(xi, np.hstack([self.w0, self.w]))
                if self.training == 'perceptron':
                    y_class = 1 if y_pred > 0 else 0
                    error = y - y_class
                    # Perceptron update rule
                    if y_class != y:
                        self.w += self.eta * error * xi[1:] 
                        self.w0 += self.eta * error   
                    missclassification_total += int(y_class != y)  
                else:
                    error = y - y_pred
                    # delta update rule
                    self.w += self.eta * error * xi[1:]         
                    self.w0 += self.eta * error             
                    missclassification_total += int(np.sign(y_pred) != y)    
                mse_total += error ** 2

            mse_epoch = mse_total / len(data)
            missclassification_epoch = missclassification_total / len(data)
            self.mse.append(mse_epoch)
            self.errors.append(missclassification_epoch)

            self.iteration += 1  # Increment the iteration counter
            if missclassification_total == 0 and self.training == 'perceptron':
                print(f'Perceptron converged after {self.iteration} epochs')
                break
            if self.iteration >= self.max_iter:
                break 
        
    def predict(self, X):
        if self.training == 'perceptron':
            return np.where(np.dot(X, self.w) + self.w0 > 0, 1, 0)
        else:
            return np.where(np.dot(X, self.w) + self.w0 > 0, 1, -1)