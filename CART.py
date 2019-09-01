import numpy as np

class TreeNode(object):
    """Tree class.
    
    (You don't need to add any methods or fields here but feel
    free to if you like. Our tests will only reference the fields
    defined in the constructor below, so be sure to set these
    correctly.)
    """
    
    def __init__(self, left, right, parent, cutoff_id, cutoff_val, prediction):
        self.left = left
        self.right = right
        self.parent = parent
        self.cutoff_id = cutoff_id
        self.cutoff_val = cutoff_val
        self.prediction = prediction


def sqsplit(xTr,yTr,weights=[]):
    """Finds the best feature, cut value, and loss value.
    
    Input:
        xTr:     n x d matrix of data points
        yTr:     n-dimensional vector of labels
        weights: n-dimensional weight vector for data points
    
    Output:
        feature:  index of the best cut's feature
        cut:      cut-value of the best cut
        bestloss: loss of the best cut
    """
    N,D = xTr.shape
    assert D > 0 # must have at least one dimension
    assert N > 1 # must have at least two samples
    if weights == []: # if no weights are passed on, assign uniform weights
        weights = np.ones(N)
    weights = weights/sum(weights) # Weights need to sum to one (we just normalize them)
   
    # TODO:
    
    index = np.argsort(xTr,axis=0)
    xs = xTr[index, np.arange(D)]
    yb = yTr.reshape(N, -1) * np.ones((N, D))
    ys = yb[index, np.arange(D)]
    wb = weights.reshape(N, -1) * np.ones((N, D))
    ws = wb[index, np.arange(D)]
    
    wy = ws * ys
    wy2 = wy * ys
    
    Q = np.cumsum(wy2, axis=0, dtype=np.double)
    P = np.cumsum(wy, axis=0, dtype=np.double)
    W = np.cumsum(ws, axis=0, dtype=np.double)
    
    losscum = - P[:-1] * P[:-1] / W[:-1] - (P[-1] - P[:-1])*(P[-1]- P[:-1]) / (1 - W[:-1]) + Q[-1]
    
    index = xs[:-1] != xs[1:]
    validLoss = losscum[index]
    if validLoss.shape[0] == 0:
        return -1, -1, -1
    loss = np.min(validLoss)
    idx = np.argwhere(np.cumsum(index) == np.argmin(validLoss) + 1)[0,0]
    i = idx // D
    f = idx % D
    cut = (xs[i, f] + xs[i + 1, f]) * 0.5
    return f, cut, loss    

    
def forest(xTr, yTr, m, maxdepth=np.inf):
    """Creates a random forest.
    
    Input:
        xTr:      n x d matrix of data points
        yTr:      n-dimensional vector of labels
        m:        number of trees in the forest
        maxdepth: maximum depth of tree
        
    Output:
        trees: list of TreeNode decision trees of length m
    """
    
    n, d = xTr.shape
    trees = []
    for i in range(m):
        random_idx = np.random.randint(0,n,size=n)
        xD = xTr[random_idx]
        yD = yTr[random_idx]
        tree = cart(xD,yD,maxdepth)
        trees.append(tree)
    
    return trees

def evalforest(trees, X, alphas=None):
    """Evaluates X using trees.
    
    Input:
        trees:  list of TreeNode decision trees of length m
        X:      n x d matrix of data points
        alphas: m-dimensional weight vector
        
    Output:
        pred: n-dimensional vector of predictions
    """
    m = len(trees)
    n,d = X.shape
    if alphas is None:
        alphas = np.ones(m) / len(trees)
            
    pred = np.zeros(n)
    
    for i in range(m):
        pred = pred + alphas[i] * evaltree(trees[i],X)
    
    return pred


def boosttree(x,y,maxiter=100,maxdepth=2):
    """Learns a boosted decision tree.
    
    Input:
        x:        n x d matrix of data points
        y:        n-dimensional vector of labels
        maxiter:  maximum number of trees
        maxdepth: maximum depth of a tree
        
    Output:
        forest: list of TreeNode decision trees of length m
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
        root = cart(x,y,maxdepth,weights)
        preds = evaltree(root,x)
        epsilon = np.sum((np.sign(preds) != y) * weights)
        
        if epsilon < 0.5:
            alpha = 0.5 * np.log((1 - epsilon) / epsilon)
            alphas.append(alpha)
            forest.append(root)
            weights = weights * np.exp(-alpha * preds * y)* 0.5 / np.sqrt(epsilon * (1 - epsilon))
        else:
            break
                    
    return forest, alphas
#</GRADED>


def evaltree(root,xTe,idx=[]):
    """Evaluates xTe using decision tree root. Same as evaltree but designed to be as efficient as possible.
    
    Input:
        root: TreeNode decision tree
        xTe:  n x d matrix of data points
        idx:  choosen indices, optional argument that might be helpful with implementation strategy
    Output:
        pred: n-dimensional vector of predictions
    """
    assert root is not None
    n = xTe.shape[0]
    pred = np.zeros(n)
    
    # TODO:
    evaltreecomp2(root,xTe,np.arange(n),pred)
       
    return pred

def evaltreecomp2(root,xTe,idx,pred):
    """
    DFS
    Evaluates xTe using decision tree root. Same as evaltree but designed to be as efficient as possible.
    
    Input:
        root: TreeNode decision tree
        xTe:  n x d matrix of data points
        idx:  choosen indices, optional argument that might be helpful with implementation strategy
        pred: n-dimensional vector of predictions
    Output:
        void
    """

    if root.cutoff_val == None:
        if root.prediction != None:              
            pred[idx] = root.prediction 
        else:          
            pred[idx] = root.parent.prediction 
        return 

 
    feature = root.cutoff_id 
    cut = root.cutoff_val
          
    lindex = idx[np.argwhere(xTe[idx, feature] <= cut).flatten()]
    
    rindex = idx[np.argwhere(xTe[idx, feature] > cut).flatten()]

    evaltreecomp2(root.left,xTe,lindex,pred) 

    evaltreecomp2(root.right,xTe,rindex,pred)


    return 

def cart(xTr,yTr,depth=np.inf,weights=None):
    """Builds a CART tree. Same as cart but designed to be as efficient as possible.
    
    The maximum tree depth is defined by "maxdepth" (maxdepth=2 means one split).
    Each example can be weighted with "weights".

    Args:
        xTr:      n x d matrix of data
        yTr:      n-dimensional vector
        maxdepth: maximum tree depth
        weights:  n-dimensional weight vector for data points

    Returns:
        tree: root of decision tree
    """
    n,d = xTr.shape
    if weights is None:
        w = np.ones(n) / float(n)
    else:
        w = weights
    head = TreeNode(None,None,None,None,None,None)
    #base case I
    if depth <= 1 or n < 2:# no cut
        head.prediction = 1.0 / np.sum(w) * np.sum(w * yTr)
        return head
        
    # recursion         
    else:

        feature, cut, bestloss = sqsplit(xTr,yTr,w)
        if feature == -1:
            return head

        Lindex = xTr[:, feature] <= cut
        Rindex = xTr[:, feature] > cut
                     
        xL = xTr[Lindex]
        yL = yTr[Lindex]
        wL = w[Lindex]
                     
        xR = xTr[Rindex]
        yR = yTr[Rindex]
        wR = w[Rindex]
        

        head.prediction = 1.0 / np.sum(w) * np.sum(w * yTr)
                
        
        head.cutoff_id = feature
        head.cutoff_val = cut
        
        head.left = cart(xL, yL, depth - 1,wL)
        
        if head.left != None:
            head.left.parent = head
            
        head.right = cart(xR, yR, depth - 1,wR)
        if head.right != None:
            head.right.parent = head
        return head
    

