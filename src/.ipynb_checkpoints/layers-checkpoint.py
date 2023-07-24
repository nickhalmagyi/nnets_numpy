import numpy as np


class Dense:
    """
    Dense Layer for a Neural Network.
    
    Key methods are [forward, backward], 
    to implement these we need to define various 
    activation functions.
    """
    
    
    def __init__(self, size, activation=None):
        
        self.N_out = size
        self.activation = activation
        
        
    def init_weights(self, N_in=None, sigmaW=None, muW=None, sigmab=None, mub=None):
        if not muW:
            muW=0
        if not sigmaW:
            sigmaW=self.N_out**(-1/2)
        if not mub:
            mub=0
        if not sigmab:
            sigmab=self.N_out**(-1)
            
        self.W = np.random.default_rng().normal(loc=muW, scale=sigmaW, size=(self.N_out, N_in))
        self.b = np.random.default_rng().normal(loc=mub, scale=sigmab, size=self.N_out)
    
    def sigmoid(self, x):
        """
        There can be numerical instabilities when computing exponentials
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        
        We want to avoid computing e^(x) when x is moderately large, as this will quickly be 
        bigger than any float. On the other hand, e^(-x) is OK when x is moderately large as 
        it goes to zero.
        """
        def sigmoid_xpos(x):
            # e^(-x) <= 1, so not large
            z = np.exp(-x)
            return 1 / (1 + z)

        def sigmoid_xneg(x):
            # e^(x) <=1, so not large
            z = np.exp(x)
            return z / (1 + z)
    
        return np.where(x >= 0, sigmoid_xpos(x), sigmoid_xneg(x))


    
    def relu(self, x):
        
        return np.maximum(x, 0)
    
    
    def sigmoid_deriv(self, x):
        
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    
    def relu_deriv(self, x):
    
        return (x > 0) * 1
    
    
    def forward(self, A_in):
        
        """
        To compute the backward pass we will need the input from the previous layer
        and the Z from the current layer and the error from the subsequent layer.
        """
        self.A_in = A_in
        batch_size = A_in.shape[0]
        Z = np.dot(A_in, self.W.T) + np.tile(self.b, [batch_size, 1])
        
        if self.activation == 'relu':
            A = self.relu(Z)
            self.fZ_deriv = self.relu_deriv(Z)
            
        elif self.activation == 'sigmoid':
            A = self.sigmoid(Z)
            self.fZ_deriv = self.sigmoid_deriv(Z)

        elif self.activation == 'id':
            A = Z

        elif self.activation == 'softmax':
            A = self.softmax(Z)
            self.fZ_deriv = self.softmax_deriv(Z)
            
        return A   


    def backward(self, A):

        return A @ self.W