import numpy as np

def ridge(w,xTr,yTr,lmbda):
    """
    INPUT:
    w     : d   dimensional weight vector
    xTr   : nxd dimensional matrix (each row is an input vector)
    yTr   : n   dimensional vector (each entry is a label)
    lmbda : regression constant (scalar)
    
    OUTPUTS:
    loss     : the total loss obtained with w on xTr and yTr (scalar)
    gradient : d dimensional gradient at w
    
    DERIVATION: 
    Let w, y and gradient g(w) all be column vectors.
    l(w) = (1/n) (X w - y)^T (X w - y) + lmbda w^T w 
    g(w) = (1/n) X^T (X w - y) + (1/n) ((X w - y)^T X)^T + lmbda w + lmbda (w^T)^T
         = (1/n) X^T (X w - y) + (1/n) X^T (X w - y) + 2 lmbda w
         = (2/n) X^T (X w - y) + 2 lmbda w
    g(w) = 2 ( X^T (X w - y) / n + lmbda w ), where lmbda = sigma^2/(n tau^2)
    """
    n, d = xTr.shape
    
    diff = np.dot(xTr, w)-yTr
    loss = np.dot(diff, diff) / n + lmbda * np.dot(w,w) #assuming lmbda means sigma^2/tau^2
    g    = 2 * (np.dot(diff, xTr) / n + lmbda * w)
    
    return loss, g


def logistic(w,xTr,yTr):
    """
    INPUT:
    w     : d   dimensional weight vector
    xTr   : nxd dimensional matrix (each row is an input vector)
    yTr   : n   dimensional vector (each entry is a label)
    
    OUTPUTS:
    loss     : the total loss obtained with w on xTr and yTr (scalar)
    gradient : d dimensional gradient at w
    
    DERIVATION:
    xi is treated as a row vector below:
    l(w) = log(1 + exp(-yi xi w)) summed over i
    g(w) = -( (yi exp(-yi xi w)) / (1 + exp(-yi xi w)) ) xi^T summed over i
         = -( yi / (exp(yi xi w) + 1) ) xi^T summed over i
    """
    n, d = xTr.shape
    
    yxw = np.dot(xTr, w) * yTr
    loss = np.sum( np.log(1 + np.exp(- yxw)) )
    g = - np.sum( (yTr / (np.exp(yxw) + 1)).reshape(n, 1) * xTr, axis=0 )
    
    return loss, g


def hinge(w,xTr,yTr,lmbda):
    """
    INPUT:
    w     : d   dimensional weight vector
    xTr   : nxd dimensional matrix (each row is an input vector)
    yTr   : n   dimensional vector (each entry is a label)
    lmbda : regression constant (scalar)
    
    OUTPUTS:
    loss     : the total loss obtained with w on xTr and yTr (scalar)
    gradient : d dimensional gradient at w
    
    DERIVATION:
    xi is treated as a row vector below:
    l(w) = max(1 - yi xi w, 0) summed over i + lmbda w^T w
    g(w) = fi(w) summed over i + 2 lmbda w, where 
    if 1 - yi xi w > 0, then fi(w) = - yi xi^T
    else fi(w) = 0
    """
    n, d = xTr.shape
    
    yxw = np.dot(xTr, w) * yTr
    loss = np.sum( np.maximum(1 - yxw, 0) ) + lmbda * np.dot(w, w)
    f = - yTr.reshape(n,1) * xTr
    g = np.sum( f[1 - yxw > 0], axis=0 ) + 2 * lmbda * w
    
    return loss, g


def adagrad(func,w,alpha,maxiter,eps,delta=1e-02):
    """
    INPUT:
    func    : function to minimize
              (loss, gradient = func(w))
    w       : d dimensional initial weight vector 
    alpha   : initial gradient descent stepsize (scalar)
    maxiter : maximum amount of iterations (scalar)
    eps     : epsilon value
    delta   : if norm(gradient)<delta, it quits (scalar)
    
    OUTPUTS:
     
    w      : d dimensional final weight vector
    losses : vector containing loss at each iteration
    """
    
    losses = np.zeros(maxiter)

    d = len(w)
    z = np.zeros(d)
    for i in range(maxiter):
        loss, g = func(w)
        losses[i] = loss
        
        if np.linalg.norm(g) < delta:
            break
        
        z += g * g
        w -= alpha * g / np.sqrt(z + eps)
        
    return w, losses


def ERM_classifier(xTr, yTr, lmbda, lossType='hinge'):
    """
    INPUT:
    xTr   : nxd dimensional matrix (each row is an input vector)
    yTr   : n   dimensional vector (each entry is a label)
    lmbda : regression constant (scalar)
    lossType : type of loss: 'ridge' | 'logistic' | 'hinge' 

    OUTPUTS:
    classifier (usage: preds = classifier(xTe))
    """

    if lossType == 'ridge':
        objective = lambda weight: ridge(weight, xTr, yTr, lmbda)
    elif lossType == 'logistic':
        objective = lambda weight: logistic(weight, xTr, yTr)
    elif lossType == 'hinge':
        objective = lambda weight: hinge(weight, xTr, yTr, lmbda) 
    else:
        raise NotImplementedError

    _, d = xTr.shape
    eps = 1e-06
    w, losses = adagrad(objective, np.random.rand(d), 1, 1000, eps)

    return ( lambda xTe: np.sign(np.dot(xTe, w)) )


def bagging(xTr, yTr, lmbda, lossType, m):
    """
    INPUT:
    m: number of classifiers in the bag
    """
    n, d = xTr.shape
    bag = []
    for i in range(m):
        random_idx = np.random.randint(0,n,size=2 * n)
        xD = xTr[random_idx]
        yD = yTr[random_idx]
        h  = ERM_classifier(xD, yD, lmbda, lossType)
        bag.append(h)

    return bag 

def evalbag(bag, xTe, alphas=None):
    m = len(bag)
    if alphas is None:
        alphas = np.ones(m) / m

    pred = np.zeros(xTe.shape[0])
    
    for i in range(m):
        pred += alphas[i] * bag[i](xTe)

    return pred
