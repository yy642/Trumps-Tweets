import numpy as np
from ERM import ERM_classifier
from kernel_svm import dualSVM
from utils import l2distance

def random_search(valSet, trainSet, getClassifier, paramInfo, maxi):
    """
    generate one combination of hyperparameters within the specified range and evaluate
    the resulting model on the validation set.

    INPUT:
    valSet: validation set, where valSet[i][0] is the features of the i-th fold
            and valSet[i][1] is the label of the i-th fold
    trainSet: training set, in the same format as valSet
    getClassifier: usage: classifier = getClassifier(a)
                          yTe = classifier(xTe)
                   where a is a tuple, a[0] is feature of the training data, a[1] is
                   the corresponding label, a[2] is the list of hyperparameters
    paramInfo: a tuple, paramInfo[0] is a list of the span of the range of hyperparameters,
               paramInfo[1] is a list of the start of the range of hyperparameters. 
               For example, if there are 2 hyperparameters, C and lmbda, C is to be searched
               in the range [10, 200] and lmbda is to searched in the range[0.2, 1] then 
               paramInfo[0] is [190, 0.8], paramInfo[1] is [10, 0.2].
    maxi: the number of folds to evaluate and average over.

    OUTPUT:
    param: the list of chosen hyperparameters
    err: the mean error on the maxi folds 
    """
    pSpan, pMin = paramInfo
    param = []

    err = 0
    for i, span in enumerate(pSpan):
        param.append( pSpan[i] * np.random.random() + pMin[i] )

    for i in range(maxi):
        classifier=getClassifier( (trainSet[i][0],trainSet[i][1],param) )
        preds = classifier(valSet[i][0])
        err += np.mean(np.sign(preds)!=valSet[i][1])
    
    err /= maxi
    print(param, err)
    return param, err


def traj(valSet, trainSet, getClassifier, paramInfo, maxi, budget, patience, best_param, best_err):
    """
    generate a trajectory of random search starting from best_param. The span is decreased 
    when a better error is found or after a number of hops without improvement.

    INPUT:
    budget: total number of hops
    patience: if the number of hops exceeds this value but no improvement is made, the span is decreased.
    best_param: the list of hyperparameters at the start of the trajectory
    best_err: error at the start of the trajectory

    OUTPUT:
    best_param: best hyperparameters along this trajectory
    best_err: best error along this trajectory
    """
    pSpan, _ = paramInfo
    pMin = []
    for i, span in enumerate(pSpan):
        m = best_param[i] - span / 2
        if m < 0:
            pMin.append(0) 
        else:
            pMin.append(m)

    count = 0
    for j in range(budget):
        paramInfo = (pSpan, pMin)
        param, err = random_search(valSet, trainSet, getClassifier, paramInfo, maxi)
        count += 1
            
        if err < best_err or count >= patience:
            if err < best_err:
                best_err = err
                best_param = param
                count = 0
                
                for i in range(len(pSpan)):
                    pSpan[i] /= 1.2
           
            else: 
                for i in range(len(pSpan)):
                    pSpan[i] /= 1.8
                
                count = 0
                print('span:',pSpan)

            pMin = []
            for i, span in enumerate(pSpan):
                m = best_param[i] - span / 2
                if m < 0:
                    pMin.append(0) 
                else:
                    pMin.append(m)
            
    return best_param, best_err

            
def getHyperparam(trainSet, valSet, modelType, paramInfo = None):
    """
    optimize hyperparameters

    INPUT:
    modelType: type of model: 'ridge' | 'hinge' | 'rbf' | 'polynomial',
               where 'rbf' and 'polynomial' refer to the kernel type of SVM

    OUTPUT:
    best hyperparameters
    """
    
    if paramInfo is None:
        param_dict = {
            #span, start of the range 
            'ridge'     : ([0.1], [0.0]), #0.1, 0.0 for simplehash, 0.002, 0.004 for one_word_bag, 0.008, 0.00 for two_word_bag 
            'hinge'     : ([10.0], [0.0]), #50, 0.0 for simplehash, 4.0, 2.0 for one_word_bag, 2.0, 0.5 for two_word_bag
            #'rbf'       : ([200.0, 200.0 / np.std(np.power(l2distance(trainSet[0][0], trainSet[0][0]), 2))], [0.0, 0.0]),
            'rbf'       : ([200.0, 0.1], [0.0, 0.0]),
            'polynomial': ([200.0, 50], [0.0, 2])
        }
        paramInfo = param_dict[modelType]

    best_param_dict = {
        'ridge'     : [0.0],  
        'hinge'     : [0.0],
        'rbf'       : [0.0, 0.0],
        'polynomial': [0.0, 2]
    }
    best_param = best_param_dict[modelType]

    #a[0]: xTr; a[1]: yTr; 
    #a[2][0]: lmbda in the case of ridge and hinge, C in the case of rbf and polynomial; 
    #a[2][1]: the other parameter in the case of rbf and polynomial
    model_dict = {
        'ridge'     : (lambda a : ERM_classifier(a[0], a[1], a[2][0], 'ridge')), 
        'hinge'     : (lambda a : ERM_classifier(a[0], a[1], a[2][0], 'hinge')), 
        'rbf'       : (lambda a : dualSVM(a[0], a[1], a[2][0], a[2][1], 'rbf')), # signature to be changed here, assume a[2][0] is C, a[2][1] is lmbda
        'polynomial': (lambda a : dualSVM(a[0], a[1], a[2][0], a[2][1], 'polynomial')) # signature to be changed here
    }
    getClassifier = model_dict[modelType]

    maxi = 5
    best_err = 1.0
    
    paramList = []
    errList = np.array([])
    for j in range(8):
        param, err = random_search(valSet, trainSet, getClassifier, paramInfo, maxi)

        for i, p in enumerate(param):
            if j == 0:
                paramList.append(np.array([p]))
            else:
                paramList[i] = np.append(paramList[i], p)
            
        errList = np.append(errList, err)
   
    seed_idx = np.argsort(errList)
    for i in range(3):

        param = []
        for j in range(len(paramList)):
            param.append(paramList[j][seed_idx[i]])

        param, err = traj(valSet, trainSet, getClassifier, paramInfo, maxi, 12, 8, param, errList[seed_idx[i]])

        if err < best_err:
            best_err = err
            best_param = param
          
    for i in range(len(paramInfo[0])):
        paramInfo[0][i] /= 4
        best_param, best_err = traj(valSet, trainSet, getClassifier, paramInfo, maxi, \
                                    16, 7, best_param, best_err)

    print("Best param, err:")
    print(best_param, best_err)
    return best_param
