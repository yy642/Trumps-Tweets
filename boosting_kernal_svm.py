import numpy as np
from cvxpy import *
from utils import l2distance

def computeK(kerneltype, X, Z, kpar=0):
    """
    function K = computeK(kernel_type, X, Z)
    computes a matrix K such that Kij=k(x,z);
    for three different function linear, rbf or polynomial.
    
    Input:
    kerneltype: either 'linear','polynomial','rbf'
    X: n input vectors of dimension d (nxd);
    Z: m input vectors of dimension d (mxd);
    kpar: kernel parameter (inverse kernel width gamma in case of RBF, degree in case of polynomial)
    
    OUTPUT:
    K : nxm kernel matrix
    """
    assert kerneltype in ["linear","polynomial","poly","rbf"], "Kernel type %s not known." % kerneltype
    assert X.shape[1] == Z.shape[1], "Input dimensions do not match"
    
    if kerneltype == "linear":
        K = np.dot(X, Z.T)
    if kerneltype == "polynomial" or kerneltype == "poly":
        K = np.power((np.dot(X, Z.T) + 1), int(kpar))
    if kerneltype == "rbf":
        K = np.exp(np.multiply(-kpar, np.power(l2distance(X, Z), 2)))
    
    return K


def dualqp(K,yTr,C, weights=[]):
    """
    function alpha = dualqp(K,yTr,C)
    constructs the SVM dual formulation and uses a built-in 
    convex solver to find the optimal solution. 
    
    Input:
        K     | the (nxn) kernel matrix
        yTr   | training labels (nx1)
        C     | the SVM regularization parameter
    
    Output:
        alpha | the calculated solution vector (nx1)
    """
    if weights == []: # if no weights are passed on, assign uniform weights
        weights = np.ones(N)
    weights = weights/sum(weights) # Weights need to sum to one (we just normalize them)


    y = yTr.flatten()
    y = y * weights
    N, _ = K.shape
    alpha = Variable(N)
    
    obj = 0.5 * quad_form(multiply(y, alpha), K) \
          - atoms.affine.sum.sum(alpha)
    constraint = [alpha >= 0, alpha <= C, atoms.affine.sum.sum(multiply(y, alpha)) == 0]
    prob = Problem(Minimize(obj), constraint)
    prob.solve()
    
    return np.array(alpha.value).flatten()


def recoverBias(K,yTr,alpha,C,weights=[]):
    """
    function bias=recoverBias(K,yTr,alpha,C);
    Solves for the hyperplane bias term, which is uniquely specified by the 
    support vectors with alpha values 0<alpha<C
    
    INPUT:
    K : nxn kernel matrix
    yTr : nx1 input labels
    alpha  : nx1 vector of alpha values
    C : regularization constant
    
    Output:
    bias : the scalar hyperplane bias of the kernel SVM specified by alphas
    """
    N = len(yTr)
    if weights == []: # if no weights are passed on, assign uniform weights
        weights = np.ones(N)
    weights = weights/sum(weights) # Weights need to sum to one (we just normalize them)


    idx = np.argmin( np.abs(alpha - C * 0.5) )
    yTr = yTr * weights
    bias = yTr[idx] - np.dot(alpha * yTr, K[:,idx])
    
    
    return bias


def dualSVM(xTr,yTr,C,lmbda,ktype,weights=[],eps=1e-10):
    """
    function classifier = dualSVM(xTr,yTr,C,ktype,lmbda);
    Constructs the SVM dual formulation and uses a built-in 
    convex solver to find the optimal solution. 
    
    Input:
        xTr   | training data (nxd)
        yTr   | training labels (nx1)
        C     | the SVM regularization parameter
        ktype | the type of kernelization: 'rbf','polynomial','linear'
        lmbda | the kernel parameter - degree for poly, inverse width for rbf
    
    Output:
        svmclassify | usage: predictions=svmclassify(xTe);
    """

    N = len(yTr)
    if weights == []: # if no weights are passed on, assign uniform weights
        weights = np.ones(N)
    weights = weights/sum(weights) # Weights need to sum to one (we just normalize them)

    
    K = computeK(ktype, xTr, xTr, lmbda)
    K = (K + K.T) / 2 + eps * np.eye(K.shape[0])
    alpha = dualqp(K,yTr,C,weights)
    b = recoverBias(K,yTr,alpha,C,weights)
    
    svmclassify = \
    lambda x: np.dot( (alpha * yTr), computeK(ktype, xTr, x, lmbda) ) + b
    
    return svmclassify

def boostSVM(x,y,maxiter=1, C=176,lmbda=2,ktype='linear'):
    """Learns a boosted decision tree.
    
    Input:
        x:        n x d matrix of data points
        y:        n-dimensional vector of labels
        maxiter:  maximum number of SVMclassifiers 
        
    Output:
        SVMclassifiers: list of SVMclassifier of length m
        alphas: m-dimensional weight vector
        
    (note, m is at most maxiter, but may be smaller,
    as dictated by the Adaboost algorithm)
    """
    assert np.allclose(np.unique(y), np.array([-1,1])); # the labels must be -1 and 1 
    n,d = x.shape
    weights = np.ones(n) / n
    preds   = None   
    forest  = []
    alphas  = []

    for m in range(maxiter):
        SVMclassifier = dualSVM(x,y,C,lmbda,ktype,weights,eps=1e-10)
        preds = SVMclassifiers(x) 
        epsilon = np.sum((np.sign(preds) != y) * weights)
        
        if epsilon < 0.5:
            alpha = 0.5 * np.log((1 - epsilon) / epsilon)
            alphas.append(alpha)
            SVMclassifiers.append(SVMclassifier)
            forest.append(root)
            weights = weights * np.exp(-alpha * preds * y)* 0.5 / np.sqrt(epsilon * (1 - epsilon))
        else:
            break
                    
    return SVMclassifiers, alphas

def evalboostingSVM(SVMclassifiers, X, alphas=None):
    """Evaluates X using a list of SVMclassifiers.
    
    Input:
        SVMclassifiers:  list of SVMclassifiers of length m
        X:      n x d matrix of data points
        alphas: m-dimensional weight vector
        
    Output:
        pred: n-dimensional vector of predictions
    """
    m = len(SVMclassifiers)
    n,d = X.shape
    if alphas is None:
        alphas = np.ones(m) / len(trees)
            
    pred = np.zeros(n)
    
    for i in range(m):
        pred = pred + alphas[i] * SVMclassifiers[i](X)
    
    return pred


