import numpy as np
import time

from src.layers import Dense
from src.functions import rmse, mse

class FeedForwardModel:
    
    def __init__(self, layer_sizes, input_size, loss_fn, alpha=0.0001):
        """
        param: layer_sizes - list of integers, representing the size of the hidden layers. 
                            The final integer is the size of the output.
               input_size - integer for the size of the input.
            loss_fn - must be in ['rmse', 'mse']
            alpha - the L2 regularization coefficient
        """
        self.layer_sizes = layer_sizes
        self.input_size = input_size
        self.layers = self._make_layers()
        self.assign_loss_fn(loss_fn)
        self.alpha = alpha
        
        self._make_input_sizes()
            
            
    def assign_loss_fn(self, loss_fn):
        if loss_fn=="rmse":
            self.loss_fn = rmse
        elif loss_fn=="mse":
            self.loss_fn = mse
        else:
            raise ValueError('loss_fn must be in the list ["rmse", "mse"]')
            
        
    def _make_input_sizes(self):
        
        self.N_ins = [self.input_size] + self.layer_sizes[:-1]
        
    def _make_layers(self):
        return [Dense(size=layer_size, activation='relu') for layer_size in self.layer_sizes[:-1]] \
            + [Dense(size=self.layer_sizes[-1], activation='id')]
    
    def initialize_weights(self):
        for layer, N_in in zip(self.layers, self.N_ins):
            layer.init_weights(N_in=N_in)

            
    def forward_pass(self, data):
        A = self.layers[0].forward(data)
        for layer in self.layers[1:]:
            A = layer.forward(A)

        y_pred = A.reshape(A.shape[0], self.layers[-1].N_out)

        return y_pred
    
    
    def backward_pass(self, y_pred, y_true, loss, learning_rate=0.1):
        
        final_layer = True
        num_samples = y_true.shape[0]
        print(f'num_samples: {num_samples}')
        for layer in self.layers[::-1]:
            if final_layer:
                if self.loss_fn==rmse:
                    layer.delta = num_samples**(-1) * loss**(-1) * (y_pred - y_true) + self.alpha * num_samples**(-1) * y_pred
                elif self.loss_fn==mse:
                    layer.delta = num_samples**(-1) * (y_pred - y_true) + self.alpha * num_samples**(-1) * y_pred
                final_layer = False
            else:
                layer.delta = np.multiply(next_layer.delta @ next_layer.W, layer.fZ_deriv)
                
            # update weights
            layer.dLdW = layer.delta.T @ layer.A_in
            # update bias
            layer.dLdb = np.sum(layer.delta, axis=0)
            
            next_layer = layer
                    
        # perform SGD on the weights
        for layer in self.layers:
            
            layer.W += - learning_rate * layer.dLdW
            layer.b += - learning_rate * layer.dLdb
            
            
        
    def train(self, data, y_true, learning_rate, epochs=3):
        
        start = time.time()
        
        for epoch_num in range(epochs):
            
            y_pred = self.forward_pass(data)
            
            num_samples = y_true.shape[0]
            loss = self.loss_fn(y_pred, y_true) + self.alpha * num_samples**(-1) * np.sum(y_pred**2) 

            self.backward_pass(y_pred=y_pred, y_true=y_true, loss=loss, learning_rate=learning_rate) 
            
            print(f"""epoch: {epoch_num}, loss: {loss}, time elapsed: {time.time()-start}\n""")