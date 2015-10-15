# SVM-Experiments
Course assignment for CS678 - Learning with Kernels at IIT Kanpur
###Data-Sets Used
These data-sets were obtained from LIBSVM library - https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

1. Leukemia 
	* # of classes: 2
	* # of data: 38 / 34 (testing)
	* # of features: 7129
2. covtype.binary
	* # of classes: 2
	* # of data: 581,012
	* # of features: 54
3. rcv1.binary
	* # of classes: 2
	* # of data: 20,242 / 677,399 (testing)
	* # of features: 47,236

## PART-1

### Checking Data-Set Format
we would be massaging out data-sets to fit into memory(the kernel matrix) and to allow fast computations. 
The data-set covtype.binary has the class labels as 1 and 2, we convert them to 1 and -1 so that the format is consistent across all the data-sets. 
After making modifications, datasets are as follows:

1. Leukemia 
	* # of classes: 2
	* # of data: 38 / 34 (testing)
	* # of features: 7129
2. covtype.binary
	* # of classes: 2
	* # of data: 581,012
	* # of features: 54
3. rcv1.binary
	* # of classes: 2
	* # of data: 20,242 / 677,399 (testing)
	* # of features: 47,236

## Doubts 
1. Is the stopping criteria required to be on the entire set or just the small subset selected ?




