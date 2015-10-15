__author__ = 'ayusek'

# Config Files for Experiments

# DATA-SETS
data_folder = "../data/"
data_set_number = 1

if(data_set_number == 1):
    train_data = data_folder + "leu.train"
    test_data =  data_folder + "leu.test"

elif(data_set_number == 2):
    train_data = data_folder + "covtype.libsvm.binary"
    test_data =  data_folder + "covtype.libsvm.binary"

elif(data_set_number == 3):
    train_data = data_folder + "covtype.libsvm.binary"
    test_data =  data_folder + "covtype.libsvm.binary"
