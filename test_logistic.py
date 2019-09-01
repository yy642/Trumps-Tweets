import numpy as np
from ERM import ERM_classifier
from dataloader import splitdata

extractfeatures = 'two_word_bag'
feature_value = 'simple'
B = 512
eps = 0.3
trainSet, valSet, _ = splitdata(['val'], extractfeatures, feature_value, B, eps)

err = 0
for i in range(5):
    classifier = ERM_classifier(trainSet[i][0], trainSet[i][1], None, 'logistic')
    preds = classifier(valSet[i][0])
    err += np.mean(np.sign(preds)!=valSet[i][1])

print(err / 5)
