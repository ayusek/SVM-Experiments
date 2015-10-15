__author__ = 'ayusek'

# Config Files for Experiments
import numpy as np
import random as rnd
rnd.seed(0)

# DATA-SETS
data_folder = "../data/"
data_set_number = 3

if(data_set_number == 1):
    train_data = data_folder + "leu.train"
    test_data =  data_folder + "leu.test"

elif(data_set_number == 2):
    train_data = data_folder + "cov.train"
    test_data =  data_folder + "cov.test"

elif(data_set_number == 3):
    train_data = data_folder + "rcv1.train"
    test_data =  data_folder + "rcv1.test"

# SVM Parameters
C = 1.0

# SMO Parameters
zero_epsilon = 10**-9
criteria = 3
verbose = False

