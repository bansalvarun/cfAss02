#!/usr/bin/python
#
# Created by Albert Au Yeung (2010)
#
# An implementation of matrix factorization
#
import numpy
import pandas as pd
from sklearn.metrics import mean_absolute_error

NUMBER_OF_USERS = 943
NUMBER_OF_ITEMS = 1682

MAX_ITERATIONS = 100
alpha = 0.0002
beta = 0.02


###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""

def matrix_factorization(R, P, Q, K, steps=MAX_ITERATIONS, alpha=alpha, beta=beta):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            print "Converged at step: ", step
            break
    return P, Q.T
def train_test_split(ratings):
    test = numpy.zeros(ratings.shape)
    train = ratings.copy()
    for user in xrange(ratings.shape[0]):
        test_ratings = numpy.random.choice(ratings[user, :].nonzero()[0],size=10,replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]

        assert(numpy.all((test*train)==0))
        return train, test
def get_nmae(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_absolute_error(pred, actual)/4.0

def get_data_matrix(path):
    # array = [[0 for j in xrange(0, NUMBER_OF_ITEMS)] for i in xrange(0, NUMBER_OF_USERS)] 
    array = numpy.zeros((NUMBER_OF_USERS, NUMBER_OF_ITEMS))
    with open(path) as file:
        for readline in file:
            temp = map(int, readline.strip().split("\t"))
            array[temp[0]-1][temp[1]-1] = temp[2]
    return array


###############################################################################

if __name__ == "__main__":
    for j in range(1,51, 5):
        print "For number of factors: "+str(j)
        for i in range(1,6):
            print "Results for u"+ str(i) + ": "
            userItemArray = get_data_matrix("ml-100k/u"+ str(i) +".base")
            testData = get_data_matrix("ml-100k/u"+ str(i) +".test")
            numberOfFactors = j
            userLatentMatrix = numpy.random.rand(NUMBER_OF_USERS, numberOfFactors)
            itemLatentMatrix = numpy.random.rand(NUMBER_OF_ITEMS, numberOfFactors)
            P, Q = matrix_factorization(userItemArray, userLatentMatrix, itemLatentMatrix, numberOfFactors, MAX_ITERATIONS, alpha, beta)
            nR = numpy.dot(P, Q.T)
            print 'User-based CF MAE: ' + str(get_nmae(nR, testData))
            print ""

