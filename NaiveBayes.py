import numpy as np
def naivebayesPY(x,y):
	"""
	naivebayesPY(x,y) returns [pos,neg]
	
	Computation of P(Y)
	Input:
	    x : n input vectors of d dimensions (nxd)
	    y : n labels (-1 or +1) (nx1)
	
	Output:
	pos: probability p(y=1)
	neg: probability p(y=-1)
	"""
	
	# add one positive and negative example to avoid division by zero ("plus-one smoothing")
	y = np.concatenate([y, [-1,1]])
	n = len(y)
	pos = (np.sum(y) / n + 1) / 2
	return (pos, 1.0 - pos)

def naivebayesPXY(x,y):
	"""
	naivebayesPXY(x,y) returns [posprob,negprob]
	
	Computation of P(X|Y)
	Input:
	    x : n input vectors of d dimensions (nxd)
	    y : n labels (-1 or +1) (nx1)
	
	Output:
	posprob: probability vector of p(x|y=1) (1xd)
	negprob: probability vector of p(x|y=-1) (1xd)
	"""
	
	# add one positive and negative example to avoid division by zero ("plus-one smoothing")
	n, d = x.shape
	x = np.concatenate([x, np.ones((2,d))])
	y = np.concatenate([y, [-1,1]])
	n, d = x.shape
	
	## fill in code here
	xpos = x * ((y == 1).reshape(-1,1))
	xneg = x * ((y == -1).reshape(-1,1))
	posprob = np.sum(xpos, axis = 0) / np.sum(xpos)
	negprob = np.sum(xneg, axis = 0) / np.sum(xneg)
	return posprob, negprob

def naivebayes(x,y,xtest):
	"""
	naivebayes(x,y) returns logratio 
	
	Computation of log P(Y|X=x1) using Bayes Rule
	Input:
	x : n input vectors of d dimensions (nxd)
	y : n labels (-1 or +1)
	xtest: input vector of d dimensions (1xd)
	
	Output:
	logratio: log (P(Y = 1|X=xtest)/P(Y=-1|X=xtest))
	"""
	
	pos,neg = naivebayesPY(x,y)
	posprob,negprob = naivebayesPXY(x,y)
	pos_and_xtest = pos * np.prod( np.power(posprob, xtest) )
	neg_and_xtest = neg * np.prod( np.power(negprob, xtest) )
	
	return np.log(pos_and_xtest / neg_and_xtest)

def naivebayesCL(x,y):
	"""
	naivebayesCL(x,y) returns [w,b]
	Implementation of a Naive Bayes classifier
	Input:
	x : n input vectors of d dimensions (nxd)
	y : n labels (-1 or +1)
	
	Output:
	w : weight vector of d dimensions
	b : bias (scalar)
	"""
	
	n, d = x.shape
	posprob,negprob = naivebayesPXY(x,y)
	pos,neg = naivebayesPY(x,y)
	w = np.log (posprob / negprob)
	b = np.log (pos / neg)
	return w,b

def classifyLinear(x,w,b=0):
	"""
	classifyLinear(x,w,b) returns preds
	
	Make predictions with a linear classifier
	Input:
	x : n input vectors of d dimensions (nxd)
	w : weight vector of d dimensions
	b : bias (optional)
	
	Output:
	preds: predictions
	"""
	
	return np.sign(np.dot(x,w) + b)

def NBclassify(x,y):
	w,b = naivebayesCL(x,y)
	return ( lambda xTe: np.sign(np.dot(xTe, w) + b) )
