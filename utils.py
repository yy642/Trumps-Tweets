import numpy as np
import datetime
oneover24 = 0.04166666666
oneover60 = 0.01666666666
def l2distance(X,Z=None):
    # function D=l2distance(X,Z)
    #
    # Computes the Euclidean distance matrix.
    # Syntax:
    # D=l2distance(X,Z)
    # Input:
    # X: nxd data matrix with n vectors (rows) of dimensionality d
    # Z: mxd data matrix with m vectors (rows) of dimensionality d
    #
    # Output:
    # Matrix D of size nxm
    # D(i,j) is the Euclidean distance of X(i,:) and Z(j,:)
    #
    # call with only one input:
    # l2distance(X)=l2distance(X,X)
    #

    if Z is None:
        Z=X;

    n,d1=X.shape
    m,d2=Z.shape
    assert (d1==d2), "Dimensions of input vectors must match!"

    def innerproduct(X,Z=None):
        # function innerproduct(X,Z)
        #
        # Computes the inner-product matrix.
        # Syntax:
        # D=innerproduct(X,Z)
        # Input:
        # X: nxd data matrix with n vectors (rows) of dimensionality d
        # Z: mxd data matrix with m vectors (rows) of dimensionality d
        #
        # Output:
        # Matrix G of size nxm
        # G[i,j] is the inner-product between vectors X[i,:] and Z[j,:]
        #
        # call with only one input:
        # innerproduct(X)=innerproduct(X,X)
        #
        if Z is None: # case when there is only one input (X)
            Z=X;
        
        G = np.dot(X, Z.T)
     
        return G
    
    preS = np.diag(innerproduct(X)).reshape(-1,1)
    preR = np.diag(innerproduct(Z)).reshape(1,-1)
    D2   = preS - 2 * innerproduct(X,Z) + preR
    D    = np.sqrt(D2)
    
    return D

def twoword(tokens):
    """
    INPUT:
    tokens: a list of word
    OUTPUT: a list of two-word combination
    """
    output = []
    for j in range(len(tokens) - 1):
        if ('http' in tokens[j] or 'http' in tokens[j + 1]):
            continue
        output.append(tokens[j : j + 2])
    tokens = [' '.join(x) for x in output]
    return tokens

def daymonthyear_dayofweek(dt):
    """
    INPUT:
    dt   : string of day/month/year(2 digits)
    
    OUTPUT:
    integer
    1: Mon, 2:Tue, 3:Wed, 4:Thur, 5:Fri, 6:Sat, 7:Sun
    """
    month,day,year=(int(x) for x in dt.split('/'))
    year = 2000 + year
    ans = datetime.date(year, month, day).toordinal() % 7
    return ans
	
def time_percentage(t):
    """
    INPUT:
    t   : string of Hour:Minute
    
    OUTPUT:
    float:
    percentage of t in 24 hours
    example: 10:00 yields 0.4167	
    """
    hour, miniute = (int(x) for x in t.split(':'))
    ans = (hour + miniute * oneover60 ) * oneover24
    return ans

def save2csv(preds):
    """
    INPUT:
    preds: the 1D array of predictions, either 1 or -1
    """
    idx = np.arange(preds.shape[0])
    result = np.concatenate((idx.reshape(-1,1), preds.reshape(-1,1)), axis=1)
    np.savetxt('prediction.csv', result, fmt='%d', delimiter=',', header='ID,Label', comments='')
