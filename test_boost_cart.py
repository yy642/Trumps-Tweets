import numpy as np
from CART import boosttree, evalforest
from dataloader import splitdata

extractfeatures = 'one_word_bag'
feature_value = 'simple'
B = 512
eps = 1.0
trainSet, valSet, _ = splitdata(['val'], extractfeatures, feature_value, B, eps)

maxiter = 50
maxdepth = 6
err = 0
for i in range(5):
    trees, alphas = boosttree(trainSet[i][0], trainSet[i][1], maxiter, maxdepth)
    err += np.mean(np.sign(evalforest(trees,valSet[i][0], alphas)) != valSet[i][1])

print(err / 5)
