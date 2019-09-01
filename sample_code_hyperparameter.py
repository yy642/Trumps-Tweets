import numpy as np
from hyperparameter import getHyperparam 
from dataloader import splitdata

extractfeatures = 'one_word_bag' #simplehash, one_word_bag, two_word_bag
feature_value = 'simple' # simple, accsimple, inverse, accinverse
B = 512 # dimensions for simplehash
eps = 1.0 # larger the eps, lower dimensional the features (for one/two word bag)
trainSet, valSet, _ = splitdata(['val'], extractfeatures, feature_value, B, eps) 

#param = getHyperparam(trainSet, valSet, 'rbf', ([200.0, 0.1], [2.0, 0.0]))# search C in initial range from 2.0 to 202.0 and lmbda from 0.0 to 0.1
param = getHyperparam(trainSet, valSet, 'rbf', ([50.0, 0.1], [0.0, 0.0]))

