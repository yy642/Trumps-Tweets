# Classify the device with which President Trump wrote each tweet

## Installation
To install `nltk`,
```
pip install -U nltk
```
To install `cvxpy` (if you have Anaconda),
```
conda install -c conda-forge lapack
conda install -c cvxgrp cvxpy
```
You can test the installation of `cvxpy` with `nose`:
```
conda install nose
nosetests cvxpy
```

## Classification
Our model that gets the highest accuracy on the public Kaggle leaderboard can be used by run
```
python test_svm.py
```
The result will be saved in `prediction.csv`.

## For developers
**sample_code.py**        : a sample code showing:
* use feature_vectorize.py to get X and Y
* use NaiveBayes.py to find w and b.

**NaiveBayes.py**         : NaiveBayes model, same as the project NB.

**feature_vectorize.py**  : Current version include
						simplehash: hash tweets into B dimension.
						bag_of_word: hash selected words.

**convert_time.py**		  : helper function used in feature_vectorize.py

**stop_word.py**		  : hepler function, a list of stop words

**feature_selection.py**  : feature selection
