import re
import csv
regex = re.compile('[^a-zA-Z]')
import nltk
from nltk.stem import PorterStemmer
ps = PorterStemmer()

import numpy as np
import sys
from scipy.io import loadmat
import time
from utils import twoword
#import stop_word
#stop_word_list = stop_word.stop_word() 

def loadtweets(extractfeatures, filename="./train.csv", do_regex=True, do_stem=True):
	'''
	process all the training set, return labels and preprocessed tokens
	INPUT:
	extractfeatures  : string, 'one_word_bag' | 'two_word_bag' | 'one_and_two_word_bag'
	                   defalut 'one_word_bag'
		           'one_word_bag'         : "abd fnu vi" --->> ["abd", "fnu", "vi"]
	                   'two_word_bag'         : "abd fnu vi" --->> ["abd fnu", fnu vi"]
		           'one_and_two_word_bag' : combine one word and two word lists

	filename         : the name of file to be processed. If open test.csv, return array of zeros as the label
        do_regex, do_stem: boolean, if true, process the text with regex and/or stem
	
	OUTPUT:
	labels           : list of lables
	alltokens        : list of tokenized and pre-processed tweets texts, the last element is date_and_time
	'''
	with open(filename, newline='') as csvfile:
	  # Skip first line (if any)
		next(csvfile, None)
		data = list(csv.reader(csvfile, delimiter=','))
	N = len(data)
	labels = np.zeros(N)
	alltokens = [] 
	if (filename == './train.csv'):
		for i in range(N):
			labels[i] = data[i][17]
	
	for i in range(N):
		tokens = data[i][1].split()
		tokens = [token.lower() for token in tokens]

		if do_regex:
			tokens = [regex.sub('', token) for token in tokens]
		if do_stem:
			tokens = [ps.stem(token) for token in tokens]

		for i in range(len(tokens)):
			if "http" in tokens[i]:
				# ignore the string after http
				tokens[i] = 'http'

		
		if extractfeatures == 'two_word_bag':# convert to two-words token
			tokens = twoword(tokens)
		elif extractfeatures == 'one_and_two_word_bag':# one and two-words token
			tokens = tokens + twoword(tokens)


		tokens.append(data[i][5])
		alltokens.append(tokens) #tweets
	print('Loaded %d input tweets.' % len(labels))
	return labels, alltokens


def find_word_freq(labels, alltokens):
	'''
	Build three dictionaries according to input labels and alltokens

	INPUT:
	labels           : size of N, a list of labels
	alltokens        : size of N, N lists of tokenized and pre-processed tweets texts.

	OUTPUT:
	total_word_dict (key: 1/2 words appeared in all tweets, value: the frequency of the word in all tweets.) 
	pos_word_dict   (key: 1/2 words appeared in pos tweets, value: the frequency of the word in pos tweets.) 
	neg_word_dict   (key: 1/2 words appeared in neg tweets, value: the frequency of the word in neg tweets.) 
	'''
	
	assert len(labels) == len(alltokens) #input labels and texts must be the same length
	total_word_dict = {}
	pos_word_dict = {}
	neg_word_dict = {}
    
	N = len(labels) 
	for i in range(N):
		tokens = alltokens[i]
		d = len(tokens)
		tokens = tokens[:-1]#skip the last element of the tokens (date and time)
		
#		if extractfeatures == 'two_word_bag':# convert to two-words token
#			tokens = twoword(tokens)
#		elif extractfeatures == 'one_and_two_word_bag':# one and two-words token
#			tokens = tokens + twoword(tokens)

		#fill in the dictonary for the tokens 
		for token in tokens:
			if token in total_word_dict:
				total_word_dict[token] += 1
			else:
				total_word_dict[token] = 1
			
			if labels[i] == 1:
				if token in pos_word_dict:
					pos_word_dict[token] += 1
				else:
					pos_word_dict[token] = 1
			else:
				if token in neg_word_dict:
					neg_word_dict[token] += 1
				else:
					neg_word_dict[token] = 1
				
	return total_word_dict, pos_word_dict, neg_word_dict

def select_word(total_word_dict, pos_word_dict, neg_word_dict, eps):
	'''
	INPUT:
	total_word_dict (key: 1/2 words appeared in all tweets, value: the frequency of the word in all tweets. 
	pos_word_dict   (key: 1/2 words appeared in pos tweets, value: the frequency of the word in pos tweets. 
	neg_word_dict   (key: 1/2 words appeared in neg tweets, value: the frequency of the word in neg tweets. 

	eps			: threshold to select word

	OUTPUT:
	word_list	: List of Strings: including all the word with 
		          ratio = (freq(word, y=1) - freq(word, y=-1)) / freq(word, y= 1 or -1)
			      larger than the eps and appear more than twice.
	df_list 	: List of Integers, the number of times a word appeared in the all tweets 
	'''

	ltot = (sum(total_word_dict.values()))
	lpos = (sum(pos_word_dict.values()))
	lneg = (sum(neg_word_dict.values()))
	word_list = []
	df_list = []
	
	idx = 0
	for word in total_word_dict.keys():
		l1 = 0
		l2 = 0
		if word in pos_word_dict:
			l1 = pos_word_dict[word]
		if word in neg_word_dict:
			l2 = neg_word_dict[word]
		ratio = (l1 / lpos - l2 / lneg) / total_word_dict[word] * ltot
		if (abs(ratio) > eps and (l1 >= 2 or l2 >= 2)):
			word_list.append(word)
			df_list.append(total_word_dict[word])

	return word_list, df_list

#l,t=loadtweets('one_word_bag', filename='./test.csv')
#print(l)		
