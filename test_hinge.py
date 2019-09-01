import numpy as np
from ERM import ERM_classifier
from dataloader import splitdata
from utils import save2csv

extractfeatures = 'one_word_bag'
feature_value = 'simple'
B = 512
eps = 0.1
eps2 = 0.2
trainSet, valSet, _ = splitdata(['val'], extractfeatures, feature_value, B, eps)

lmbda = 4.840903908178895 
err = 0
train_err = 0
for i in range(5):
    classifier = ERM_classifier(trainSet[i][0], trainSet[i][1], lmbda, 'hinge')
    preds = classifier(valSet[i][0])
    cur_err = np.mean(np.sign(preds)!=valSet[i][1])
    err += cur_err
    print('fold {}: {}'.format(i, cur_err))
    preds = classifier(trainSet[i][0])
    train_err += np.mean(np.sign(preds)!=trainSet[i][1])

print("train err:", train_err / 5)
print("val err:", err / 5)

trainSet, _, testSet = splitdata(['test'], extractfeatures, feature_value, B, eps)
classifier = ERM_classifier(trainSet[0][0], trainSet[0][1], lmbda, 'hinge')
preds = classifier(testSet)
save2csv(preds)
