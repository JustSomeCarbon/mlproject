# mlproject
Project for CSE 589 Machine Learning

## Quick Start Guide
The main files for the project are `svm.py` and `naive_bayes.py`.
In order for the code to run, the user must be in an anaconda environment.

## Naive Bayes
The Naive Bayes file can take some parameters to specify what it should run.
The user can specify whether to use `bag-of-words` or `TF-IDF` as the pre-training
for the dataset. The user can specify whether to use `multinomial` or `gaussian` 
Naive Bayes model.  
The following are some examples that can be used:
~~~
python3 naive_bayes.py -p tf-idf -m gaussian
python3 naive_bayes.py -p bow -m multinomial
~~~