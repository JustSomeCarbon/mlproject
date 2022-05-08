#
# file: naive_bayes.py
# description: This file contains the source code for the Naive Bayes model
#       implementation. The program expects the input to be already cleaned.
#

import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import argparse
import pandas as pd
import numpy as np
from bow import *

# increase epoch to find better mean accuracy
epoch = 20
run_val = 0

# argparse definitions
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pre_train", default='bow', choices=['bow', 'tf-idf'])
ap.add_argument("-m", "--model", default='multinomial', choices=['multinomial', 'gaussian'])

args = vars(ap.parse_args())
pre = args['pre_train']
mod = args['model']


# global filenames
nb_bow_filename = "nb_bow.out"
nb_tf_filename = "nb_tfidf.out"
# global file variables
if pre == 'bow':
    nb_bow_file = open(nb_bow_filename, "w+")
else:
    nb_tfidf_file = open(nb_tf_filename, "w+")


#
# MAIN FUNCTION
#
def main():
    print("Running Naive Bayes Sentiment Analysis Model Construction")

    dataset = pd.read_csv("cleandata.csv")

    # divide the dataset
    text_data = dataset["Text"]
    label_data = dataset["Emotion"]
    total_entries = len(text_data)

    if pre == 'bow':
        nb_bow_func(text_data, label_data)
    else:
        nb_tfidf_func(text_data, label_data)


#
# nb_bow_func - overhead function that creates a bag of words object from
#       the text data. The bow is then utilized in the construction and
#       evaluation of a Naive Bayes model over the data.
#
def nb_bow_func(text_data, label_data):
    print("Generating bag-of-words from dataset...")
    # call for bag of words over data
    bag_data = Bow(text_data, label_data)
    #print(bag_data.bag)
    print("bag-of-words object initialized!")

    split_values = [0.2, 0.25, 0.3, 0.35, 0.4]

    # create the Naive Bayes model
    if mod == 'mulitnomial':
        print("Constructing Multinomial Naive Bayes model...")
        nb_bow_file.write("Multinomial Naive Bayes using BOW::\n")
        for split_value in split_values:
            for i in range(epoch):
                nb_model_bow(label_data, bag_data, split_value)
    else:
        print("Constructing Gaussian Naive Bayes model...")
        nb_bow_file.write("Gaussian Naive Bayes using BOW::\n")
        for split_value in split_values:
            for i in range(epoch):
                gaus_nb_model_bow(label_data, bag_data, split_value)


#
# nb_tfidf - overhead function that transforms the given text data into tf-idf
#       vector representations. Those vectors are then utilized in the construction
#       and evaluation of a Naive Bayes movel over the data.
#
def nb_tfidf_func(text_data, label_data):
    print("Transforming dataset with TF-IDF...")

    split_values = [0.2, 0.25, 0.3, 0.35, 0.4]

    if mod == 'multinomial':
        print("Constructing Multinomail Naive Bayes model...")
        nb_tfidf_file.write("Multinomial Naive Bayes using TF-IDF::\n")
        for split_value in split_values:
            for i in range(epoch):
                nb_model_tfidf(label_data, text_data, split_value)
    else:
        print("Constructing Gaussian Naive Bayes model...")
        nb_tfidf_file.write("Gaussian Naive Bayes using TF-IDF::\n")
        for split_value in split_values:
            for i in range(epoch):
                gaus_nb_model_tfidf(label_data, text_data, split_value)


#
# nb_model_bow - takes the label data from the dataset and a bag-of-words object
#       to create a multinomial Naive Bayes model
#
def nb_model_bow(label_data, bag_data, splt_val):
    # split the training and testing dataset
    x_train, x_test, y_train, y_test = train_test_split(bag_data.bag, label_data, test_size=splt_val)
    
    # create the model
    mnd = MultinomialNB()
    mnd.fit(x_train, y_train)

    # generate the accuracy score for the model
    naive_predict = mnd.predict(x_test)
    acc = accuracy_score(y_test, naive_predict)

    print(" Model evaluation run test split:", splt_val)
    print("  Accuracy of Multinomial Naive Bayes using BOW:", (acc*100))
    write_out(nb_bow_filename, acc, splt_val)


#
# gaus_nb_model_bow - takes the label data from the dataset and a bag-of-words object
#       to create a Gaussian Naive Bayes model.
#
def gaus_nb_model_bow(label_data, bag_data, splt_val):
    # split the training and testing dataset
    x_train, x_test, y_train, y_test = train_test_split(bag_data.bag, label_data, test_size=splt_val)
    
    # create the model
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)

    # generate the accuracy score for the model
    naive_predict = gnb.predict(x_test)
    acc = accuracy_score(y_test, naive_predict)

    print(" Model evaluation run test split:", splt_val)
    print("  Accuracy of Gaussian Naive Bayes using BOW:", (acc*100))
    write_out(nb_bow_filename, acc, splt_val)


#
# nb_model_tfidf - takes the label data from the dataset and the text data to
#       perform TF-IDF over the dataset and creates a multinomial Naive Bayes model
#
def nb_model_tfidf(label_data, text_data, splt_val):
    x_train, x_test, y_train, y_test = train_test_split(text_data, label_data, test_size=splt_val)
    
    # create the term frequency vectorizer object
    #tf_vect = CountVectorizer()
    tf_vect = TfidfVectorizer()
    # trained vectorizer
    tf_train = tf_vect.fit_transform(x_train)
    tf_test = tf_vect.transform(x_test)
    #print("train shape: {}".format(tf_train.shape))
    #print("test shape: {}".format(tf_test.shape))

    # create the Naive Bayes model
    mnb = MultinomialNB()
    mnb.fit(tf_train, y_train)

    # determine the accuracy of the model
    naive_predict = mnb.predict(tf_test)
    acc = accuracy_score(y_test, naive_predict)

    print(" Model evaluation run test split:", splt_val)
    print("  Accuracy of Multinomial Naive Bayes using tf-idf:", (acc*100))
    write_out(nb_tf_filename, acc, splt_val)


#
# gaus_nb_model_tfidf - takes the label data from the dataset and the text data to
#       perform TF-IDF over the dataset and creates a Gaussian Naive Bayes model
#
def gaus_nb_model_tfidf(label_data, text_data, splt_val):
    x_train, x_test, y_train, y_test = train_test_split(text_data, label_data, test_size=splt_val)
    
    # create the term frequency vectorizer object
    tf_vect = TfidfVectorizer()
    # trained vectorizer
    tf_train = tf_vect.fit_transform(x_train)
    tf_test = tf_vect.transform(x_test)

    # create the Naive Bayes model
    gnb = GaussianNB()
    gnb.fit(tf_train.todense(), y_train)

    # determine the accuracy of the model
    naive_predict = gnb.predict(tf_test.todense())
    acc = accuracy_score(y_test, naive_predict)

    print(" Model evaluation run test split:", splt_val)
    print("  Accuracy of Gaussian Naive Bayes using tf-idf:", (acc*100))
    write_out(nb_tf_filename, acc, splt_val)


#
# write_out - write the accuracy out the the appropriate file
#
def write_out(filename, acc_val, splt_val):
    if (filename == nb_bow_filename):
        acc = str(run_val) + "::" + str(splt_val) + ": Accuracy Val: " + str(acc_val*100) + "\n"
        nb_bow_file.write(acc)
    elif (filename == nb_tf_filename):
        acc = str(run_val) + "::" + str(splt_val) + ": Accuracy Val: " + str(acc_val*100) + "\n"
        nb_tfidf_file.write(acc)


if __name__ == '__main__':
    main()