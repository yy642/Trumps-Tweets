import numpy as np
from kernel_svm import dualSVM
from dataloader import splitdata
from utils import save2csv

extractfeatures = 'one_word_bag'
feature_value = 'simple'
B = 512
eps = 1.0
trainSet, valSet, _ = splitdata(['val'], extractfeatures, feature_value, B, eps)

err = 0
for i in range(5):
    classifier = dualSVM(trainSet[i][0], trainSet[i][1], 1.7403839415716706, 0.08235396195138298, 'rbf')
    preds = classifier(valSet[i][0])
    err += np.mean(np.sign(preds)!=valSet[i][1])

print(err / 5)

trainSet, _, testSet = splitdata(['test'], extractfeatures, feature_value, B, eps)
classifier = dualSVM(trainSet[0][0], trainSet[0][1], 1.7403839415716706, 0.08235396195138298, 'rbf')
preds = classifier(testSet)
save2csv(np.sign(preds))
