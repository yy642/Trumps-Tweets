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


def dualqp(K,yTr,C):
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
    y = yTr.flatten()
    N, _ = K.shape
    alpha = Variable(N)
    
    obj = 0.5 * quad_form(multiply(y, alpha), K) \
          - atoms.affine.sum.sum(alpha)
    constraint = [alpha >= 0, alpha <= C, atoms.affine.sum.sum(multiply(y, alpha)) == 0]
    prob = Problem(Minimize(obj), constraint)
    prob.solve()
    
    return np.array(alpha.value).flatten()


def recoverBias(K,yTr,alpha,C):
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

    idx = np.argmin( np.abs(alpha - C * 0.5) )
    bias = yTr[idx] - np.dot(alpha * yTr, K[:,idx])
    
    
    return bias


def dualSVM(xTr,yTr,C,lmbda,ktype,eps=1e-10):
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
    
    K = computeK(ktype, xTr, xTr, lmbda)
    K = (K + K.T) / 2 + eps * np.eye(K.shape[0])
    alpha = dualqp(K,yTr,C)
    b = recoverBias(K,yTr,alpha,C)
    
    svmclassify = \
    lambda x: np.dot( (alpha * yTr), computeK(ktype, xTr, x, lmbda) ) + b
    
    return svmclassify
