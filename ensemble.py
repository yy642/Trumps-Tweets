import numpy as np
import feature_selection as fs
import feature_vectorize as fv

def splitdata(split):
    """
    INPUT:
        split = ['val' | 'test']
        other inputs: see feature_selection.py, feature_vectorize.py
    
    OUTPUT:
        (trainSet, valSet, testSet), where trainSet and valSet are lists of tuples and testSet is
        a tensor.
        valSet: validation set, where valSet[i][0] is the features of the i-th fold
                and valSet[i][1] is the label of the i-th fold, non-empty only if 
                'val' is in split
        trainSet: training set, in the same format as valSet(only 1 fold if 'val' is not in split),
                  always returned. When 'val' is not in split, this contains all the data from 
                  train.csv.
        testSet: test set features (N * D tensor), non-empty only if 'test' is in split and 'val'
                 is not in split
    
    """
    #load raw data and do the pre-processing and tokenization
    labels, tokens = fs.loadtweets(extractfeatures, filename="./train.csv")	

    trainSet, valSet, testSet = [], [], np.array([])
    if 'val' not in split:
        if extractfeatures != 'simplehash':
            d1, d2, d3 = fs.find_word_freq(labels, tokens)
            word_list, df_list = fs.select_word(d1, d2, d3, eps)
        else:
            word_list, df_list = [], []
        trainSet.append((fv.loaddata(tokens, extractfeatures, word_list, df_list, feature_value, B, eps), labels))
        if 'test' in split:
            _, testTokens = fs.loadtweets(extractfeatures, filename="./test.csv")
            testSet = fv.loaddata(testTokens, extractfeatures, word_list, df_list, feature_value, B, eps)
        return trainSet, valSet, testSet
    else:
        # cross validation
        N = len(tokens)
        tokens = np.array(tokens)
        # put random shuffle outside
        idx_list = np.arange(N)
        np.random.shuffle(idx_list)
        preValSet = [
            (tokens[idx_list[:N//5]], labels[idx_list[:N//5]]),
            (tokens[idx_list[N//5 : (2 * N)//5]], labels[idx_list[N//5 : (2 * N)//5]]),
            (tokens[idx_list[(2 * N)//5 : (3 * N)//5]], labels[idx_list[(2 * N)//5 : (3 * N)//5]]),
            (tokens[idx_list[(3 * N)//5 : (4 * N)//5]], labels[idx_list[(3 * N)//5 : (4 * N)//5]]),
            (tokens[idx_list[(4 * N)//5:]], labels[idx_list[(4 * N)//5:]])
        ]
        preTrainSet = [
            (tokens[idx_list[N//5:]],labels[idx_list[N//5:]]),
            (np.concatenate((tokens[idx_list[:N//5]], tokens[idx_list[(2 * N)//5:]])), np.concatenate((labels[idx_list[:N//5]], labels[idx_list[(2 * N)//5:]]))),
            (np.concatenate((tokens[idx_list[:(2 * N)//5]], tokens[idx_list[(3 * N)//5:]])), np.concatenate((labels[idx_list[:(2 * N)//5]], labels[idx_list[(3 * N)//5:]]))),
            (np.concatenate((tokens[idx_list[:(3 * N)//5]], tokens[idx_list[(4 * N)//5:]])), np.concatenate((labels[idx_list[:(3 * N)//5]], labels[idx_list[(4 * N)//5:]]))),
            (tokens[idx_list[:(4 * N)//5]], labels[idx_list[:(4 * N)//5]])
        ]
        return preTrainSet, preValSet, testSet

def ensenmble(preTrainSet, preValSet, testSet):
    for i in range(5): 
        if extractfeatures != 'simplehash':
            d1, d2, d3 = fs.find_word_freq(preTrainSet[i][1], preTrainSet[i][0])
            word_list, df_list = fs.select_word(d1, d2, d3, eps)
        else:
            word_list, df_list = [], []
        trainSet.append((fv.loaddata(preTrainSet[i][0], extractfeatures, word_list, df_list, feature_value, B, eps), preTrainSet[i][1]))
        valSet.append((fv.loaddata(preValSet[i][0], extractfeatures, word_list, df_list, feature_value, B, eps), preValSet[i][1]))


    return trainSet, valSet, testSet
            
        
