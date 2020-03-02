import numpy as np
# define layers    
class linear_layer(object):
    '''
    implement z = WX + b
    '''
    def __init__(self, in_dim, out_dim, step_size = 0.1, l2_reg = 0):
        '''
        in_dim - input dimension, feature dimention of X
        out_dim - output dimension, feature dimention of z
        step_size - update step size for gradient discent
        l2_reg - l2 norm regularization for W. Opitmizing loss funtion J'(W, X, y) = J(W, X, y) + (l2_reg/2) * |w|_2^2
        '''
        self.in_dim = in_dim
        self.out_dim = out_dim
        ## random initiate parameters, sample from normal disrtibution
        self.W = np.random.randn(self.out_dim, self.in_dim) + 0.01
        self.b = np.random.randn(self.out_dim, 1)+ 0.01
#         self.b = np.zeros((self.out_dim, 1))
        self.step = step_size
        self.l2reg = l2_reg
        
    def forward_prop(self, X):
        self.m = X.shape[1] # number of samples 
        out = np.dot(self.W, X) + self.b
        return out
    
    def backward_prop(self, dz, X):  
        # dz: gradient of output
        # X : input
        self.dW = np.dot(dz, X.T) / self.m + (self.l2reg/self.m) * self.W
        self.db = np.sum(dz) / self.m 
        # gradient of the input (last layer activation)
        da = np.dot(self.W.T, dz)
        return da
        
    def update(self):
        self.W -= self.step * self.dW
        self.b -= self.step * self.db
        

class logistic_regression_head(object):
    '''
        output layer of this binary classification problem.
        take the last hidden layer as input and output a single prediction value.
    '''
    def __init__(self, in_dim, step_size, l2_reg = 0):
        ## random initiate parameters, sample from normal disrtibution
        self.step = step_size
        self.w = np.random.randn(1, in_dim) * 0.01
        self.b = np.random.randn() * 0.01
        self.l2reg = l2_reg
        
    def forward_prop(self, X):
        self.m = X.shape[1] # number of samples 
        self.z = np.dot(self.w, X) + self.b
        self.A = sigmoid(self.z)  # shape(1, m)
        return self.A
        
    def backward_prop(self, dz, X):
        ## gradient of weights
        self.dw = np.dot(dz, X.T) / self.m + (self.l2reg/self.m) * self.w
        self.db = np.sum(dz) / self.m 
        ## gradient of input
        dx = np.dot(self.w.T, dz)
        return dx
        
    def update(self):
        self.w -= self.step * self.dw
        self.b -= self.step * self.db
        
        

class Dropout(object):
    '''
    dropout layer. Randomly zeroes some of the elements of the input tensor with probability p. 
    Each samples and features will be zeroed out independently.
    '''
    def __init__(self, rate = 0.5):
        '''
        X - input data
        rate - drop out rate
        '''
        self.p = rate

    def forward_prop(self, A):
        self.mask = self._construct_mask(self.p, A.shape)
        return np.multiply(A, self.mask) / (1-self.p)

    def backward_prop(self, dA):
        return np.multiply(dA, self.mask) / (1-self.p)


    def _construct_mask(self, rate, size):
        '''
        helper method for constructing mask
        '''
        mask = np.random.uniform(0, 1, size)
        mask = mask > rate
        return mask
    
## define functions
def sigmoid(z):
    return 1/(1+ np.exp(-z))

def cross_entropy_cost(A, y):
    '''
    A: prediction; 
    y: true value
    '''      
    m = len(y) # number of samples
    return -(np.dot(np.log(A), y.T) + np.dot(np.log(1 - A), (1 - y.T)))/m

def relu(z):
    return np.maximum(z, 0)

def grad_relu(z):
    '''
    gradient of relu given input z 
    a = relu(z)
    z: input
    '''
#     dz = np.array(da, copy = True)
    dz = np.ones_like(z)
    dz[z <= 0] = 0
    return dz
   
def grad_tanh(a):
    '''
    gradient of tanh given (input z or) gradient of output a
    a = tanh(z)
    '''
    dz = 1 - np.square(a)
    return dz