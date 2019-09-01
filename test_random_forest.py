import numpy as np
from CART import forest, evalforest
from dataloader import splitdata

extractfeatures = 'one_word_bag'
feature_value = 'accsimple'
B = 512
eps = 1.4
trainSet, valSet, _ = splitdata(['val'], extractfeatures, feature_value, B, eps)

num_trees = 100
err = 0
for i in range(5):
    trees = forest(trainSet[i][0], trainSet[i][1], num_trees)
    cur_err = np.mean(np.sign(evalforest(trees,valSet[i][0])) != valSet[i][1])
    err += cur_err
    print('fold {}: {}'.format(i, cur_err))

print(err / 5)
