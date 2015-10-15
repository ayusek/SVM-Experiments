__author__ = 'ayusek'

from ex_config import *

import svmutil
import numpy as np

Y, X = svmutil.svm_read_problem(train_data)

Y = np.asarray(Y)

ymin = min(Y)
ymax = max(Y)

Y[Y == ymin] = -1
Y[Y == ymax] = 1

