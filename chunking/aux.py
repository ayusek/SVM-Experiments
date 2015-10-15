__author__ = 'ayusek'

from config import *
import numpy as np
import cvxopt

def get_H(X , Y):
    K = X*np.transpose(X)
    H = K.todense()

    Y = np.matrix(np.reshape(Y, (len(Y),1)))
    return np.matrix(np.array(H)*np.array(Y * np.transpose(Y)))

def get_Ld(alpha , H):
    return (float)(0.5*np.dot(alpha.transpose() , np.dot(H , alpha)) - sum(alpha))

# Returns the gradient vector
def get_grad_Ld(alpha , H):
    return np.dot(H , alpha) - 1

def distance_Function(alpha , X, Y , C):
    W = np.transpose(alpha*Y)*X
    sv = np.where(np.logical_and(alpha > zero_epsilon , alpha < C  - zero_epsilon))[0]
    W0 = np.mean(Y[sv]) - np.mean((X*W)[sv])
    #W0 = np.mean(Y[sv]) - np.mean(W*X[sv,:] , axis = 0)
    return (lambda x : W*(x.transpose()) + W0)




