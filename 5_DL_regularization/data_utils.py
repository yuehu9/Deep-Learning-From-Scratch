import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

def plot_decision_boundary(model, X, y):
    '''The function for plotting the decision function takes as arguments an anonymous function 
    used to generate the predicted labels, and applies the function to the training data.

    plot_decision_boundary(lambda x: clf.predict(x), train_X, C)

    The function calls the predict method from the class
    LogisticRegressionCV that implements logistic regression in scikit-learn.

    ---------------------------------------------------------------

    predict(X)[source]
    Predict class labels for samples in X.

    Parameters: 
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Samples.

    Returns:    
    C : array, shape = [n_samples]

    Predicted class label per sample.

    ---------------------------------------------------------------

    In order to use the same function for plotting the decision boundary
    for your neural network model, you need a function for predicting the labels.
    Because of the matrix dimensions, you need a separate function that takes as input
    the training examples using a matrix of shape (m,n) and outputs the labels
    as a 1-d array.

    For example, you can implement a function:

    predict_plot(parameters,X) 

    where paramaters are the weights and biases of the neural network model and 
    X is the training data size of shape (m,n).

    Then, you can plot the boundary using 

    plot_decision_boundary(lambda x: predict_plot(parameters, x), train_X, C)
    '''
    
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()

# Load 2D data from sklearn
def load_moons(plot = True):  
    # Generate the data
    N = 200
    X, Y = sklearn.datasets.make_moons(n_samples=N, noise=.3)
    X, Y = X.T, Y.reshape(1, Y.shape[0])

    # For plotting colors
    C = np.ravel(Y)
    
    # Visualize the data
    if plot:
    	plt.scatter(X[0, :], X[1, :], c=C, s=40, cmap=plt.cm.Spectral);
    
    return X, Y

