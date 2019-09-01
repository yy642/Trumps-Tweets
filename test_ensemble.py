import numpy as np
from ERM import ERM_classifier
from dataloader import splitdata
from utils import save2csv
from scipy import stats
#from NaiveBayes import NBclassify
from kernel_svm import dualSVM
from CART import forest, evalforest
from CART import boosttree
from NaiveBayes import NBclassify
#method, extractfeatures, feature_value, hyperpara, eps, B, error
argments=[
#    ['rbf', 'simplehash', 'simple', [175.9767703699569, 0.06862774937411875], 0.1], # 0.14326723882805564
    ['ridge', 'one_and_two_word_bag', 'simple', [0.0219624524125561], 0.1], # 0.14600262123197902 
#    ['rbf', 'one_word_bag', 'simple', [96.43246619634907, 0.04814201461714812], 0.1], # 0.1386758550712383
#    ['rbf', 'one_word_bag', 'simple',[1.7403839415716706, 0.08235396195138298], 1.0], # 0.14327569441508475
    ['hinge', 'one_word_bag', 'simple',[4.840903908178895], 0.1], # 0.14691159683761046
#    ['randomforest', 'one_word_bag', 'simple', [50], 0.1], # 0.15055595484716527
#    ['boosttree', 'one_word_bag', 'simple', [50, 6], 0.1], # 0.15335052636029256
#    ['rbf', 'one_word_bag', 'accsimple',[3.9769660245213054, 0.1022640666575626], 0.5],#0.14507673445228936
#    ['rbf', 'one_word_bag', 'accsimple',[4.326070498603125, 0.03532842488832063], 1.25],#0.14781634464972732
#    ['rbf', 'one_word_bag', 'accsimple',[99.33136780650678, 0.0545784486580033], 0.1],#0.146915824631125
#    ['rbf', 'one_and_two_word_bag', 'accsimple',[185.65843311359043, 0.002551351593068753], 1.0],#0.1496638904155921 
    ['rbf', 'one_and_two_word_bag', 'accsimple',[5.94808028522554, 0.012421808360679885], 1.25],#0.14417198664017253
#    ['rbf', 'one_and_two_word_bag', 'accsimple',[24.26711317670318, 0.034526328194645205], 0.1],#0.14420158119477447
    ['NB', 'one_word_bag', 'accsimple',[],1.4],#16.15904959201793
    ['NB', 'one_two_word_bag', 'accsimple',[],1.3]#17.262926478670785
]





def ensemble(method,extractfeatures, feature_value, hyperpara, eps, B=512):
    trainSet, valSet, testSet = splitdata(['test'], extractfeatures, feature_value, B, eps)
    preds=[]
    for i in range(1):
        if method == 'hinge' or method == 'ridge' or method == 'logistic': 
            classifier = ERM_classifier(trainSet[i][0], trainSet[i][1], hyperpara[0], method) # lamda
            pred = np.sign(classifier(testSet))
        elif method == 'NB': 
            classifier = NBclassify(trainSet[i][0], trainSet[i][1])
            pred = np.sign(classifier(testSet))
        elif method == 'polynomial' or method == 'rbf' or method == 'linear':
            classifier = dualSVM(trainSet[i][0], trainSet[i][1], hyperpara[0], hyperpara[1], method) # hyperpara[0]:C hyperpara[1]: lambda
            pred = np.sign(classifier(testSet))
        elif method == 'boosttree':
            trees, alphas = boosttree(trainSet[i][0], trainSet[i][1], hyperpara[0], hyperpara[1]) #hyperpara[0]: maxiter,  hyperpara[1]: maxdepth
            pred = np.sign(evalforest(trees,testSet, alphas)) 
        elif method == 'randomforest':
            trees = forest(trainSet[i][0], trainSet[i][1], hyperpara[0]) # hyperpara[0]:number of trees
            pred = np.sign(evalforest(trees,testSet)) 
        preds.append(pred)
    return preds



err = 0

m = len(argments)
preds = [np.array([]) for i in range(1)]
for j in range(m):
    arg = argments[j] 
    preds_classifier = ensemble(*arg)
    for i in range(1):
        if j == 0:
            preds[i] = preds_classifier[i].reshape(-1,1)
        else:
            preds[i] = np.concatenate((preds[i], preds_classifier[i].reshape(-1, 1)), axis=1)
np.set_printoptions(threshold=np.inf)        

#trainSet, valSet, _ = splitdata(['val'], 'simplehash', 'simple')
#for i in range(len(preds[0])):
#    print(preds[0][i], valSet[0][1][i])
for j in range(1):
    pred = stats.mode(np.sign(preds[j]), axis=1)[0].reshape(-1)
    count = stats.mode(np.sign(preds[j]), axis=1)[1].reshape(-1)
#    err += np.mean(pred != valSet[j][1]) 

#print("val err:", err / 5)

#trainSet, _, testSet = splitdata(['test'], extractfeatures, feature_value, B, eps)
#classifier = ERM_classifier(trainSet[0][0], trainSet[0][1], lmbda, 'hinge')
#preds = classifier(testSet)
save2csv(pred)
