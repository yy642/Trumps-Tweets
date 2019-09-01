import numpy as np
#from CART import boosttree, evalforest
from dataloader import splitdata
from boosting_kernal_svm import evalboostingSVM,boostSVM 
from kernel_svm import dualSVM

extractfeatures = 'one_word_bag'
feature_value = 'simple'
B = 512
eps = 0.1
trainSet, valSet, _ = splitdata(['val'], extractfeatures, feature_value, B, eps)


err1 = 0
err2 = 0
for i in range(5):
    classifier1 = dualSVM(trainSet[i][0], trainSet[i][1], 176, 2,'linear')

#def boostSVM(x,y,maxiter=1, C=176,lmbda=2,ktype='linear'):
    classifier2, alphas = boostSVM(trainSet[i][0], trainSet[i][1], maxiter=1, C=176, lmbda=2,ktype='linear')
    preds1 = classifier1(valSet[i][0])

#def evalboostingSVM(SVMclassifiers, X, alphas=None):
    preds2 = evalboostingSVM(classifier2, valSet[i][0], alphas) 
    erri1 += np.mean(np.sign(preds1)!=valSet[i][1])
    erri2 += np.mean(np.sign(preds2)!=valSet[i][1])

    print(err1 / 5)
    print(err2 / 5)
