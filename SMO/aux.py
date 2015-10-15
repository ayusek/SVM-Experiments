__author__ = 'ayusek'

import numpy as np
import random as rnd
import cvxopt
from config import *

# Auxillary Functions
def kernel(x1,x2):
    return (x1*x2.transpose())[0,0]

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


def perform_initial_chunking(X , Y , chunk_size , C):

    index_set = rnd.sample(range(len(Y)), chunk_size)

    S_X = X[index_set]
    S_Y = Y[index_set]
    S_H = get_H(S_X , S_Y)

    assert(chunk_size == len(S_Y))


    P = cvxopt.matrix(S_H , tc = 'd')
    Q = cvxopt.matrix(-1*np.ones(chunk_size) , tc = 'd')
    G = cvxopt.matrix(np.concatenate((-1*np.identity(chunk_size) , np.identity(chunk_size)), axis = 0 ) , tc = 'd')
    h = cvxopt.matrix(np.concatenate((np.zeros(chunk_size), C*np.ones(chunk_size))) , tc = 'd' )
    A = cvxopt.matrix(np.reshape(S_Y.transpose() , (1 , chunk_size)) , tc = 'd')
    b = cvxopt.matrix([0] , tc = 'd')

    cvxopt.solvers.options['show_progress'] = False
    output =  cvxopt.solvers.qp(P,Q,G,h,A,b)

    alpha_part  = np.array(output['x'])
    alpha = np.zeros(len(Y))

    for i in range(chunk_size):
        if(alpha_part[i][0] > zero_epsilon and alpha_part[i][0] < C - zero_epsilon):
            alpha[index_set[i]] = alpha_part[i][0]

    return np.array(alpha)

def is_Ipositive(alphai , yi , C):
    return (alphai < C - zero_epsilon and yi == 1) or (alphai > C-zero_epsilon and alphai < C + zero_epsilon and yi == -1)


def is_Inegative(alphai , yi, C):
    return (alphai < C - zero_epsilon and yi == -1) or (alphai > C-zero_epsilon and alphai < C + zero_epsilon and yi == 1)

def get_new_alpha_indices(alpha, X, Y, C , H):
    grad_alpha = get_grad_Ld(alpha , H)
    comparing_argument = np.array((-1*np.array(Y)*np.array(grad_alpha))[0])
    max_positive = -10**6
    min_negative = 10**6

    max_index = -1
    min_index = -1

    for i in range(len(Y)):
        if(is_Ipositive(alpha[i], Y[i], C)):
            if(comparing_argument[i] > max_positive):
                max_positive = comparing_argument[i]
                max_index = i

        if(is_Inegative(alpha[i] , Y[i], C)):
            if(comparing_argument[i] < min_negative):
                min_negative = comparing_argument[i]
                min_index = i

    assert (max_index != -1)
    assert (min_index != -1)
    return (max_index , min_index)

def stopping_criteria(alpha , X , Y , C , H):
    if(criteria == 3):
        comparator = -1*np.array(Y)*np.array(get_grad_Ld(alpha, H))[0]

        assert(len(comparator) == len(alpha))

        max = -10**6
        min = 10**6

        for i in range(len(comparator)):
            if(is_Ipositive(alpha[i] , Y[i] , C)):
                if(comparator[i] > max):
                    max = comparator[i]


            if(is_Inegative(alpha[i] , Y[i] , C)):
                if(comparator[i] < min):
                    min = comparator[i]

            assert(is_Ipositive(alpha[i] , Y[i] , C) != is_Inegative(alpha[i] , Y[i] , C))

        if(verbose) :print max
        if(verbose) :print min
        if(verbose) :print "max - min is:" , max - min
        print "max - min is:" , max - min
        return (max - min) <= zero_epsilon

def distance_Function(alpha , X, Y , C):
    W = np.transpose(alpha*Y)*X
    sv = np.where(np.logical_and(alpha > zero_epsilon , alpha < C  - zero_epsilon))[0]
    W0 = np.mean(Y[sv]) - np.mean((X*W)[sv])
    #W0 = np.mean(Y[sv]) - np.mean(W*X[sv,:] , axis = 0)
    return (lambda x : W*(x.transpose()) + W0)


