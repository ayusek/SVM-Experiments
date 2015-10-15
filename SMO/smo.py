__author__ = 'ayusek'

from numpy import *
from config import *
from sklearn.datasets import load_svmlight_file
from aux import *

def SMO(X , Y , initial_chunk_size, C):
    iteration = 0

    alpha = perform_initial_chunking(X, Y, initial_chunk_size, C)
    # print alpha

    assert(len(alpha) , len(Y))

    # TODO : VERIFY THAT THIS IS A VALID ALPHA

    H = get_H(X, Y)

    # I pray to god that my implementation so far is correct _/\_

    while(not stopping_criteria(alpha , X , Y , C , H)):
        iteration += 1
        print "*************** ITERATION ", iteration, "*************** "
        alpha_1_index , alpha_2_index  =  get_new_alpha_indices(alpha, X, Y, C, H)

        print "Indices Chosen are:", alpha_1_index , alpha_2_index

        alpha_1_old = alpha[alpha_1_index]
        alpha_2_old = alpha[alpha_2_index]

        if(verbose) :print "Old Values :", alpha_1_old , alpha_2_old

        y1 = Y[alpha_1_index]
        y2 = Y[alpha_2_index]

        if(y1 != y2):
            h = min(C , alpha_2_old - alpha_1_old + C)
            l = max(0 , alpha_2_old - alpha_1_old)

        else:
            h = min(C , alpha_1_old + alpha_2_old)
            l = max(0, alpha_1_old + alpha_2_old - C)


        # Calculating K
        K11 = kernel(X[alpha_1_index] , X[alpha_1_index])
        K22 = kernel(X[alpha_2_index] , X[alpha_2_index])
        K12 = kernel(X[alpha_1_index] , X[alpha_2_index])

        assert(kernel(X[alpha_1_index] , X[alpha_2_index]) == kernel(X[alpha_2_index], X[alpha_1_index]))

        K = (K11 + K22 - 2*K12)

        assert(K >= 0 )

        #Calculating E
        V = y1*(alpha_1_old*y1 + alpha_2_old*y2)

        K1 = 0
        for i in range(len(alpha)):
            if(i != alpha_1_index and i!= alpha_2_index):
                K1 += alpha[i]*Y[i]*kernel(X[alpha_1_index] , X[i])

        K2 = 0
        for i in range(len(alpha)):
            if(i != alpha_1_index and i!= alpha_2_index):
                K2 += alpha[i]*Y[i]*kernel(X[alpha_2_index] , X[i])

        E = K1 - K2 + alpha_1_old*(K11 - K12) + y2*alpha_2_old*(K12 - K22) - y1 + y2
        alpha_2_new = alpha_2_old + (y2*E)/K

        #Update the alpha vector
        assert(h >= l)

        if(alpha_2_new > h):
            if(verbose) :print "Upper Bound Clash"
            alpha_2_new = h

        if(alpha_2_new < l):
            if(verbose) :print "Lower Bound Clash"
            alpha_2_new = l

        if(y1 == y2):
            if(verbose) :print "Y Match :", y1 == y2
            alpha_1_new = alpha_1_old + alpha_2_old - alpha_2_new
        else:
            if(verbose) :print "Y Match :", y1 == y2
            alpha_1_new = alpha_1_old - alpha_2_old + alpha_2_new

        alpha[alpha_1_index] = alpha_1_new
        alpha[alpha_2_index] = alpha_2_new

        if(verbose) :print "New Values are :",alpha_1_new , alpha_2_new

        assert(alpha[alpha_2_index] <= C and alpha[alpha_2_index] >= 0)
        assert(alpha[alpha_1_index] <= C and alpha[alpha_1_index] >= 0)

    return alpha

        # Reiterate

print "Training Data File :" , train_data
print "Test Data File :", test_data
print "====================================== Starting Optimization : ======================================"

X, Y = load_svmlight_file(train_data)
X_test, Y_test = load_svmlight_file(test_data)

C_best = 0
error_best = 100

for C in np.logspace(-3,5,num=9,base=10):
    alpha = SMO(X, Y , 200 , C)
    discriminator = distance_Function(alpha , X , Y , C)

    Prediction = discriminator(X)
    Prediction[Prediction >= 0 ] = 1
    Prediction[Prediction < 0] = -1

    error = sum(abs(Prediction[Prediction != Y])) / len(Y)
    print "Error on the training set  is : ", error*100, "% ", "Accuracy on the training set is : ", (1-error)*100

    Prediction = discriminator(X_test)
    Prediction[Prediction >= 0 ] = 1
    Prediction[Prediction < 0] = -1

    error = sum(abs(Prediction[Prediction != Y_test])) / len(Y_test)
    print "Error on the test set  is : ", error*100, "% ", "Accuracy on the test set is : ", (1-error)*100

    if(error < error_best):
        error_best = error
        C_best = C

print "######################################################"
print "Best Error on the test set  is : ", error_best*100, "% ", "Accuracy on the test set is : ", (1-error_best)*100, "For C = ", C