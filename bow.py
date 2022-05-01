#
# file: bow.py
# description: This file contains the definition of the Bag-of-Words
#       class for the Naive Bayes program.
#

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


#
# Class to handle the Bag-of_words information
#
class Bow:
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.bag, self.uniq_words, self.vocab_index = self.bow_vectorize(texts)


    #
    # bow_vectorizer - fits the text data passed in to a bag-of-words
    #       model and returns the information vectors
    #       returned:: (bow_vector, unique_words_vector, index_vocabulary_vector)
    #
    def bow_vectorize(self, texts):
        vectorizer = CountVectorizer()

        # fit the texts to a bag-of-words model
        bag = vectorizer.fit_transform(texts.to_numpy())
        # get unique words/tokens found in all texts
        uniq = vectorizer.get_feature_names()
        # associate inidices with each unique word
        vocab = vectorizer.vocabulary_

        return (bag.toarray(), uniq, vocab)