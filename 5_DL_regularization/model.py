import numpy as np
import matplotlib.pyplot as plt
import sklearn

import importlib, layers
importlib.reload(layers)
from layers import *

from sklearn.metrics import accuracy_score, f1_score
# define model
class deep_feedforward_network(object):
    def __init__(self, X_train, y_train, hidden_list, activation = 'reLu', drop_out = 0, step_size = 0.1, l2_reg = 0, max_iter = 1000,tol = 1e-2, random_seed = 0, verbal = True):
        '''
        X_train - training data
        y_train - training label
        hidden_list - specify width for each layer. e.g., [5,5] means two hidden layer, each with 5 hidden features.
        activation - numlinear activation function, support reLu and tanh
        step_size - update step size for gradient discent
        max_iter - maximum iteration
        l2_reg - l2 norm regularization for W. Opitmizing loss funtion J'(W, X, y) = J(W, X, y) + (l2_reg/2) * |w|_2^2
        random_seed - random seed for reproductbility
        verbal - print training losses or not
        drop_out - drop out rate for all hidden layers. e.g., 0 means do not drop out, 0.7 means drop 70% neurons. 
        '''
        self.verbal = verbal
        self.seed = random_seed
#         np.random.seed(self.seed)
        self.step = step_size
        self.tol = tol
        self.activation = activation
        self.max_iter = max_iter
        self.X = X_train
        self.y = y_train
        self.n0 = X_train.shape[0] # input feature size
        self.m = X_train.shape[1] # number of samples
        self.layer_width = hidden_list
        self.number_layers = len(hidden_list)
        self.l2reg = l2_reg
        self.p_dropout = drop_out
        
        # LAYERS
        self.layers = []
        self.dropout_layers = []
        # first hidden layer
        self.layers.append(linear_layer(self.n0, self.layer_width[0], self.step, self.l2reg)) 
        # middle layers
        for i in range(1, self.number_layers):
            self.layers.append(linear_layer(self.layer_width[i-1], self.layer_width[i], self.step, self.l2reg))
        # output layer
        self.output_layer = logistic_regression_head(self.layer_width[self.number_layers -1], self.step)
        # dropout layer for each hidden layers
        for i in range(self.number_layers):
            self.dropout_layers.append(Dropout(self.p_dropout))
        
    ### training functions #####
    def forward_prop(self, X, eval = False):
        '''Given training data A, forward propogate though all layers and output prediction value
        X - input
        eval - in evaluation mode or not. training mode would need more storage space for back prop.
        '''
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
    
    def backward_prop(self):
        '''
        backward propogation, store gradients in the process
        '''
        dzL = self.aL - self.y
        da = self.output_layer.backward_prop(dzL, self.ai[-1])
        for i in range(self.number_layers):
            da = self.dropout_layers[-1-i].backward_prop(da)
            z = self.zi[-1-i]
            a = self.ai[-2-i]
            dz = np.multiply(da, grad_relu(z))
            layer = self.layers[-1-i]
            da = layer.backward_prop(dz, a)
    
    def update(self):
        '''
        update weights based on stored gradients.
        '''
        self.output_layer.update()
        for i in range(self.number_layers):
            layer = self.layers[-1-i]
            layer.update()
                       
    def fit(self):
        '''
        wrap forward prop, back prop and update, fit the data. record the cost vs. iteration
        '''
        self.costs = np.zeros(self.max_iter) # record costs
        self.costs[:] = np.nan
        for i in range(self.max_iter):
            A = self.forward_prop(self.X)
            self.costs[i] = cross_entropy_cost(A, self.y)
            self.backward_prop()
            self.update()
            
            if i % 100 == 0 and self.verbal:
                print("Cost after iteration %i : %f" %(i, self.costs[i]))
                
            if self.costs[i] < self.tol:
                print("Cost less than %e after iteration %i : %f" %(self.tol, i, self.costs[i]))
                break
                
    def plot_curve(self):
        '''
        plot cost v.s. iteration curve
        '''
        plt.plot(self.costs)
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.title('learning rate: ' + str(self.step))
        plt.show()
        
    ### testing function ###
    def predict(self, X_test, sample_as_row = False):
        '''
        predict the lable of datapoint X_test
        X_test - test data
        output:
        label, 0 or 1
        '''
        if sample_as_row:
            X_test = X_test.T
        # if sigmoid value >= 0.5, possitive
        A = self.forward_prop(X_test, eval = True)
        return A >= 0.5
    
    def evaluate(self, test_X, test_y):
        '''
        evaluate the performance of the model given a test set
        test_X - test set features
        test_y - test set labels
        output: print accuracy and F1 socre of model prediction, 
                alse print training accuracy and F1 socre as comparison.
        '''
        # training score
        train_y_pred = self.predict(self.X).flatten()
        train_acc = accuracy_score(self.y, train_y_pred)
        train_f1 = f1_score(self.y, train_y_pred)
        print("Training accuracy: %.3f   F1 score: %.3f" %(train_acc, train_f1))
        # testing score
        test_y_pred = self.predict(test_X).flatten()
        test_acc = accuracy_score(test_y_pred, test_y)
        test_f1 = f1_score(test_y_pred, test_y)
        print("Testing accuracy: %.3f    F1 score: %.3f" %(test_acc, test_f1))    