
from torch import optim
import torch
import torch.nn as nn
import torch.nn.init as init

class ThreeLayerNet(nn.Module):
    def __init__(self, input_size, output_size, nh_1, nh_2, init_low, init_high):
        super(ThreeLayerNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, nh_1),
            nn.Sigmoid(),
            nn.Linear(nh_1, nh_2),
            nn.Sigmoid(),
            nn.Linear(nh_2, output_size),
            nn.Identity()
        )
        self.init_low = init_low
        self.init_high = init_high
        self.initialize_weights()


    def forward(self, x):
        return self.net(x)
    
    def predict(self, x):
        return self.forward(x)
    
    def initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init.uniform_(layer.weight, self.init_low, self.init_high)
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)

def fit(x_train, y_train, x_val, y_val, model, optimizer,
        loss_func, epochs = 1000, epsilon = 0.001, early_stop_crit = 15):
    validation_errors = []
    training_errors   = []
    best_val_error = 100000000

    for epoch in range(epochs):
        model.train() 
        optimizer.zero_grad()
        predictions = model(x_train).squeeze()
        loss = loss_func(predictions, y_train)
        loss.backward()
        optimizer.step()
        
        model.eval() 
        with torch.no_grad():
            val_predictions = model(x_val).squeeze()
            val_loss = loss_func(val_predictions, y_val)
        
        training_errors.append(val_loss.item())
        validation_errors.append(loss.item())
        
        if abs(val_loss.item() - best_val_error) < epsilon:
            no_improvement += 1
        else:
            best_val_error = val_loss.item()
            no_improvement = 0 
        
        if no_improvement >= early_stop_crit:
            # Compare changes in "early_stop_crit" # of consecutive val errors
            print(f'Early stopping at epoch = {epoch}')
            print(f'Training Error = {round(loss.item(), 4)}, Validation Error = {round(val_loss.item(), 4)}')
            break
    return validation_errors, training_errors, epoch