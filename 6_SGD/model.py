import numpy as np
import matplotlib.pyplot as plt
import sklearn

import importlib, layers
importlib.reload(layers)
from layers import *

from sklearn.metrics import accuracy_score, f1_score
# define model
class deep_feedforward_network(object):
    def __init__(self, n0, hidden_list, activation = 'reLu', drop_out = 0, step_size = 0.1, l2_reg = 0, random_seed = 0, momentum= False, beta = 0.9):
        '''
        n0 - input feature size
        hidden_list - specify width for each layer. e.g., [5,5] means two hidden layer, each with 5 hidden features.
        activation - numlinear activation function, support reLu and tanh
        step_size - update step size for gradient discent
        max_iter - maximum iteration
        l2_reg - l2 norm regularization for W. Opitmizing loss funtion J'(W, X, y) = J(W, X, y) + (l2_reg/2) * |w|_2^2
        random_seed - random seed for reproductbility
        verbal - print training losses or not
        drop_out - drop out rate for all hidden layers. e.g., 0 means do not drop out, 0.7 means drop 70% neurons. 
        '''
        self.seed = random_seed
#         np.random.seed(self.seed)
        self.step = step_size
        self.activation = activation
        self.layer_width = hidden_list
        self.n0 = n0
        self.number_layers = len(hidden_list)
        self.l2reg = l2_reg
        self.p_dropout = drop_out
        
        # LAYERS
        self.layers = []
        self.dropout_layers = []
        # first hidden layer
        self.layers.append(linear_layer(self.n0, self.layer_width[0], self.step, self.l2reg,   momentum = momentum, beta = beta)) 
        # middle layers
        for i in range(1, self.number_layers):
            self.layers.append(linear_layer(self.layer_width[i-1], self.layer_width[i], self.step, self.l2reg,  momentum = momentum, beta = beta))
        # output layer
        self.output_layer = logistic_regression_head(self.layer_width[self.number_layers -1], self.step,  momentum = momentum, beta = beta)
        # dropout layer for each hidden layers
        for i in range(self.number_layers):
            self.dropout_layers.append(Dropout(self.p_dropout))
        
    ### training functions #####
    def forward_prop(self, X, eval = False):
        if not eval:
            # in training mode, store intermediate variables for back prop
            self.zi = []
            self.ai = []
            a = X
            self.ai.append(a)
            for i in range(self.number_layers):
                layer = self.layers[i]
                z = layer.forward_prop(a)
                if self.activation == 'tanh':
                    a = np.tanh(z)
                else:
                    a = relu(z)
                a = self.dropout_layers[i].forward_prop(a)
                self.zi.append(z)
                self.ai.append(a)
            self.aL = self.output_layer.forward_prop(a)
            return self.aL
        else:
            # in evaluatation mode, do not store variables, and do not do drop out
            a = X
            for i in range(self.number_layers):
                layer = self.layers[i]
                z = layer.forward_prop(a)
                if self.activation == 'tanh':
                    a = np.tanh(z)
                else:
                    a = relu(z)     
            self.aL = self.output_layer.forward_prop(a)
            return self.aL
    
    def backward_prop(self, y_true):
        dzL = self.aL - y_true
        da = self.output_layer.backward_prop(dzL, self.ai[-1])
        for i in range(self.number_layers):
            da = self.dropout_layers[-1-i].backward_prop(da)
            z = self.zi[-1-i]
            a = self.ai[-2-i]
            dz = np.multiply(da, grad_relu(z))
            layer = self.layers[-1-i]
            da = layer.backward_prop(dz, a)
    
    def update(self):
        self.output_layer.update()
        for i in range(self.number_layers):
            layer = self.layers[-1-i]
            layer.update()
                       
    def fit(self, Xs, y_true):
        '''
        fit a given minibatch Xs, ys
        return cost
        '''
        ys = self.forward_prop(Xs)
        cost = cross_entropy_cost(ys, y_true)
        self.backward_prop( y_true)
        self.update()
        return cost

            
        
    ### testing function ###
    def predict(self, X_test, sample_as_row = False):
        if sample_as_row:
            X_test = X_test.T
        # if sigmoid value >= 0.5, possitive
        A = self.forward_prop(X_test, eval = True)
        return A >= 0.5
    
    def evaluate(self,train_X, train_y, test_X, test_y):
        # training score
        train_y_pred = self.predict(train_X).flatten()
        train_acc = accuracy_score(train_y, train_y_pred)
        train_f1 = f1_score(train_y, train_y_pred)
        print("Training accuracy: %.3f   F1 score: %.3f" %(train_acc, train_f1))
        # testing score
        test_y_pred = self.predict(test_X).flatten()
        test_acc = accuracy_score(test_y_pred, test_y)
        test_f1 = f1_score(test_y_pred, test_y)
        print("Testing accuracy: %.3f    F1 score: %.3f" %(test_acc, test_f1))    