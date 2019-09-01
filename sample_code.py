import NaiveBayes as NB
import feature_vectorize as fv
import feature_selection as fs
import time
import numpy as np
def newtest(extractfeatures, feature_value = None, B = 512, eps = 0.1):
	"""
	sample code for (1) load raw data, get labels and all pre-processed and tokeniezd data
					(2) split labels and all data into two sets: train and validation
					(3) if not extractfeatures='simplehash', build dictionaries using traning data
					(4) if not extractfeatures='simplehash', select a list of important words using dictonaries
					(5) vectorize the features

	extractfeatures  : string, 'one_word_bag' | 'two_word_bag' | 'one_and_two_word_bag'
	                   defalut 'one_word_bag'
			   'one_word_bag'         : "abd fnu vi" --->> ["abd", "fnu", "vi"]
	                   'two_word_bag'         : "abd fnu vi" --->> ["abd fnu", fnu vi"]
		           'one_and_two_word_bag' : combine one word and two word lists

	feature_value    : string, 'simple' | 'accsimple' | 'inverse' | 'accinverse'
                           defalt value: 'simple'

	B                : Integer, If you choose simplehash, B is the feature dimesion, defalt value is 512. 
			    	   If you choose other method, B is not used.
	eps              : double, if you choose one_word_bag or two_word_bag, eps would be the threshold for 
                	   selecting important words among all training data. Note that the value of eps would automatically 
			   set the feature dimension, which is the number of words selected. defalut value is 0.1, this would 
	   	           result in ~1000 words in one_word_bag. Higher eps means lower feature dimension.  
	"""

	#load raw data and do the pre-processing and tokenization
	labels, alltokens = fs.loadtweets(extractfeatures, filename="./train.csv")	

	#split into training and validation data
	k=50
	trtokens = alltokens[k:]
	YTr = labels[k:]

	vatokens = alltokens[:k] 
	YVa = labels[:k]

	print("Method =", extractfeatures)
	if extractfeatures != 'simplehash':	
		#build dict
		d1,d2,d3 = fs.find_word_freq(YTr, trtokens)
		#select words
		word_list,df_list = fs.select_word(d1,d2,d3,eps)
		print(word_list)
		print(df_list)

		print("select words using dicts bulit with eps=", eps)
		print("feature dimension=", len(word_list))
	else:
		print("feature dimension =", B)
		word_list,df_list = [],[]
	
	print("vectorize training data and validation data via", extractfeatures, "and", feature_value)
	XTr = fv.loaddata(trtokens, extractfeatures, word_list, df_list, feature_value, B, eps)
#	XVa = fv.loaddata(vatokens, extractfeatures, word_list, df_list, feature_value, B, eps)

	#print("compute w and b ")
	w,b = NB.naivebayesCL(XTr,YTr)

	print('Training error: %.2f%%' % (100 *(NB.classifyLinear(XTr, w, b) != YTr).mean()))
	print('Validation error: %.2f%%' % (100 *(NB.classifyLinear(XVa, w, b) != YVa).mean()))

#using one_word_bag and simple value 
extractfeatures='one_word_bag'
feature_value=None
#the following two should be the same
newtest(extractfeatures, feature_value ,eps=2.5)
newtest(extractfeatures, 'simple',eps=2.5)


#using one_word_bag and inverse value
#extractfeatures='one_and_two_word_bag'
#feature_value='inverse'
#newtest(extractfeatures, feature_value ,eps=0.5)


#using two_word_bag and inverse value
#extractfeatures='one_word_bag'
#feature_value='accsimple'
##feature_value='simple'
#list1= np.arange(1.0,2.0,0.1)
#for eps in list1:
#	newtest(extractfeatures, feature_value ,eps=float(eps))



