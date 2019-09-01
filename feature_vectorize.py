import csv
import stop_word
stop_word_list = stop_word.stop_word()
import nltk
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import re
regex = re.compile('[^a-zA-Z!]')

import numpy as np
import sys
from scipy.io import loadmat
import time

import feature_selection as fs 
from utils import twoword
from utils import daymonthyear_dayofweek 
from utils import time_percentage 


def bag_of_word(tokens, word_list, df_list, feature_value=None):
	'''
	INPUT:
	text	          : the input(i.e. tweets) to extract features
	word_list         : the list of words selected from the training set 
	df_list 	  : List of Integers, the number of how many time does a word appeared in training set 
	feature_value     : String, what kind of value to put into the bucket, defalut value is "simple"
		            "simple"     : put 1
			    "accsimple"  : put the frequency of the word in this tweet
			    "inverse"    : put 1.0 / number of times the feature appeared in all the tweets (Reason for
			    "accinverse" : put accumulated 1.0 / number of times the feature appeared in all the tweets 
			    (Reason for doing this is: if a word is frequent, it is less important)
	
	OUTPUT:
	v		  : the vectorized feature 
	'''
	N = len(word_list)
	v = np.zeros(N)

	if feature_value == 'inverse':
		for token in tokens:
		#skip unseen/unselected words
			if token in word_list:
				position = word_list.index(token)
				v[position] = 1.0 / np.log(df_list[position])
	elif feature_value == 'accinverse':
		for token in tokens:
			#skip unseen/unselected words
			if token in word_list:
				position = word_list.index(token)
				v[position] += 1.0 / np.log(df_list[position])
	elif feature_value == 'accsimple':
		for token in tokens:
			#skip unseen/unselected words
			if token in word_list:
				position = word_list.index(token)
				v[position] += 1
	else:
		for token in tokens:
			#skip unseen/unselected words
			if token in word_list:
				position = word_list.index(token)
				v[position] = 1

	return v


def simplehash(tokens, B):
	'''
	*** Simple hash ***
    *** Only hash the tweets ***
	INPUT:
	text			  : the input(i.e. tweets) to extract features
	B				  : dimensionality of feature space
	
	OUTPUT:
	v				  : the vectorized feature 
	'''
	v = np.zeros(B)
	for token in tokens:
		v[hash(token) % B] = 1
	return v


def loaddata(data, extractfeatures=None, word_list=[], df_list=[], feature_value = 'simple', B = 512, eps = 0.1):
	'''
	INPUT:
	extractfeatures  : string, 'one_word_bag' | 'two_word_bag' | 'one_and_two_word_bag'
			   'one_word_bag'         : "abd fnu vi" --->> ["abd", "fnu", "vi"]
	                   'two_word_bag'         : "abd fnu vi" --->> ["abd fnu", fnu vi"]
		           'one_and_two_word_bag' : combine one word and two word lists
	                    defalut 'one_word_bag'

					
	data              : List: size of N, N lists of tokenized and pre-processed tweets texts.
	word_list         : a list of selected words, defalt None
	df_list           : the frequency of selected words, defalt None
	feature_value     : String, what kind of value to put into the bucket
		           "simple": put 1
			   "inverse" : put 1.0 / number of times the feature appeared in all the tweets 
			   (Reason: if a word is frequent, it is less important)
			   "loginverse" : np.log(1.0 / number of times the feature appeared in all the tweets)


	B		 : Integer, dimensionality of feature space for tweets, not including tweet time
			   only used if choose 'simplehash',defalut = 512
	eps		 : Double, the threshold for selecting word
			   only used if not choose 'simplehash', defalt = 0.1

	
	OUTPUT:
	xs 	         : N x d array, the vectorized feature, where d=B if choose 'simplehash', d=len(word_list) if not 
	'''
	N = len(data)
	xtime = np.zeros((N, 2))
	#get and vectorize the tweet time
	for i in range(N):
		dt = data[i][-1].split()
		date = dt[0]
		time = dt[1]
		xtime[i, 0] = daymonthyear_dayofweek(date)
		xtime[i, 1] = time_percentage(time)

    # vectorize the tweets:	
	if extractfeatures == "simplehash":
		xtext = np.zeros((N, B))
		for i in range(N):
			xtext[i, :] = simplehash(data[i][:-1], B)
	else:
		assert len(word_list) != 0 # must input a word_list
		dim = len(word_list)
		xtext = np.zeros((N, dim))
		for i in range(N):
			xtext[i, :] = bag_of_word(data[i][:-1], word_list, df_list, feature_value)
	xs = np.concatenate((xtext, xtime), axis = 1)	
	return xs

#load raw data
#labels, alltokens = fs.loadtweets(filename="./train.csv")	
#split into training and validation data
#trtokens = alltokens[20:]
#trlabels = labels[20:]

#vatokens = alltokens[:20] 
#valabels = labels[:20]
#build dict using traning data
#d1,d2,d3 = fs.find_word_freq(trlabels, trtokens, word_combinations=None)
#select words using dicts bulit and eps
#wl,dl = fs.select_word(d1,d2,d3, eps=1.5)
#vectorize training data and validation data
#extractfeatures = 'one_word_bag'
#feature_value = 'inverse'
#XTr = loaddata(trtokens, extractfeatures , wl, dl, feature_value, B = 512, eps = 0.1)
#YTr = trlabels
#XVa = loaddata(vatokens, extractfeatures,  wl, dl,  feature_value, B = 512, eps = 0.1)
#YVa = valabels
#print(XVa)
#print(date_and_time)
#N = len(labels)

#trtokens = alltokens[:20]
#trlabels = labels[:20]
#d1,d2,d3=find_word_freq(trlabels, trtokens)
#wl,dl = select_word(d1,d2,d3, 1.5)

