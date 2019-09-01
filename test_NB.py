import NaiveBayes as NB
import feature_vectorize as fv
import feature_selection as fs
import time
import numpy as np

#from ERM import ERM_classifier
from dataloader import splitdata

extractfeatures = ['one_word_bag', 'two_word_bag', 'one_and_two_word_bag']
extractfeaturesname = ['one', 'two', 'one_and_two']
feature_value = ["simple","accsimple","inverse","accinverse"]
eps = np.arange(0.1, 2.6, 0.1)
B = 512

for i in range(len(extractfeatures)):
	for j in range(len(feature_value)):
		output=open( "NB_" + extractfeaturesname[i] + "_" + feature_value[j] + ".txt",'w')
		besterr = np.inf
		besteps = np.inf 
		for k in range(len(eps)):
			trainSet, valSet, _ = splitdata(['val'], extractfeatures[i], feature_value[j], B, eps[k])
			trainerr = 0
			valerr = 0
			for ii in range(5):
				w,b = NB.naivebayesCL(trainSet[ii][0],trainSet[ii][1])
#				trainerr += 100 *(NB.classifyLinear(trainSet[ii][0], w, b) != trainSet[ii][1]).mean()
				valerr += 100 *(NB.classifyLinear(valSet[ii][0], w, b) != valSet[ii][1]).mean()

			output.write(str(round(eps[k],2)) + " " + str(valerr/5))
			output.write("\n")
			if valerr < besterr:
				besterr = valerr
				besteps = eps[k] 
		print(besteps)
		output.write("best eps=" + str(round(besteps,2)) + " best val err=" + str(besterr / 5))
