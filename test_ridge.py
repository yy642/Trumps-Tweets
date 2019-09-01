import numpy as np
from ERM import ERM_classifier, bagging, evalbag
from dataloader_tmp import splitdata

extractfeatures = 'one_two_word_bag'
feature_value = 'simple'
B = 512
eps = 0.1
eps2 = 0.2
trainSet, valSet, _ = splitdata(['val'], extractfeatures, feature_value, B, eps, eps2)

lmbda = 0.018442197196142596
m = 100
err = 0
train_err = 0
for i in range(5):
    classifier = ERM_classifier(trainSet[i][0], trainSet[i][1], lmbda, 'ridge')
    preds = classifier(valSet[i][0])
    #classifiers = bagging(trainSet[i][0], trainSet[i][1], lmbda, 'ridge', m)
    #preds = evalbag(classifiers, valSet[i][0])

    err += np.mean(np.sign(preds)!=valSet[i][1])

    #preds = classifier(trainSet[i][0])
    #preds = evalbag(classifiers, trainSet[i][0])

    #train_err += np.mean(np.sign(preds)!=trainSet[i][1])

    
print("train err:", train_err / 5)
print("val err:", err / 5)


