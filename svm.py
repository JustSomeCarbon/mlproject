#
# file: svm.py
# description: This file contains the source code for the Support Vector Machine
#       model implementation. The program expects the input to be already cleaned.
#

#import matplotlib.pyplot as plt
import argparse
from numpy import vectorize
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
#import numpy as np
import pandas as pd
#import seaborn as sns
#from sympy import deg
#import tf_idf

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#
# Classes possible: Sadness, Anger, Love, Surprise, Fear, and Happiness (6)
#

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pre_train", default='tf-idf', choices=['bow', 'tf-idf'])
ap.add_argument("-m", "--model", default='linear', choices=['linear', 'rbf', 'poly'])


args = vars(ap.parse_args())
pre = args['pre_train']
mod = args['model']

# global output filenames
svm_linear_filename = "svm_linear.out"
svm_rbf_filename = "svm_rbf.out"
svm_poly_filename = "svm_poly.out"

# global file variables
if mod == 'linear':
    linear_file = open(svm_linear_filename, "w+")
elif mod == 'rbf':
    rbf_file = open(svm_rbf_filename, "w+")
else:
    poly_file = open(svm_poly_filename, "w+")

epoch = 5
run_val = 0

def main():
    # read in the cleaned data
    dataset = pd.read_csv("cleandata.csv")


    # DIVIDE THE DATASET
    text_data = dataset["Text"]
    label_data = dataset["Emotion"]
    total_entries = len(text_data)
    print("Total text entries:", total_entries)

    # TF (term frequency) - how common a specific word is
    # TF(t) = (number of times term t appears in the document) / (total number of terms in the document)

    # IDF (inverse document frequency) - how unique or rare the word is
    # TDF(t) = log_e(total number of documents / number of documents with term t in it)

    # TF-IDF weight = TF * IDF  (product of the two quantities)

    #text_term_freq = tf_idf.term_frequency(tf_idf.tok_frequency(text_data)) # frequency score of local terms : array (TF)
    #frequency_count = tf_idf.word_per_doc(tf_idf.tok_frequency(text_data))  # frequency score of global terms : dictionary
    #idf = tf_idf.determine_idf(text_term_freq, frequency_count)             # idf score determined from frequency : array (IDF)

    # generate the tf-idf product
    #tf_idf_score = tf_idf.generate_tf_idf(text_term_freq, idf)  # TF-IDF score

    #text_score = tf_idf.score_texts(tf_idf_score)               # Score array for each text in the data

    # training and testing split
    # x - random assortment of text_data
    # y - associated random assortment of label_data
    #x_train, x_test, y_train, y_test = train_test_split(text_score.reshape(-1, 1), label_data.to_list(), test_size=0.3)
    # access value : y_train[0]
    # access at index : y_train.index[0]

    #generate_SVM(x_train, x_test, y_train, y_test)

    # generate model training and testing data and store the results
    # in the output files
    run_val = 1
    split_values = [0.2, 0.25, 0.3, 0.35, 0.4]
    print("Training/Testing split values:", split_values)
    print("Building model and testing at each split epoch number:", epoch)

    if mod == 'linear':
        print("Building and testing SVM with linear kernel...")
        for split_val in split_values:
            print(" Testing with split value:", split_val)
            for i in range(epoch):
                if pre == 'tf-idf':
                    print("  Building linear SVM with TF-IDF...")
                    svm_linear(text_data, label_data, split_val)
                elif pre == 'bow':
                    print("  Building linear SVM with BOW...")
                    svm_linear_bow(text_data, label_data, split_val)
                run_val += 1
        print("Testing complete")
        linear_file.close()

    if mod == 'rbf':
        #run_val = 1
        print("Building and tsting SVM with RBF kernel...")
        for split_val in split_values:
            print(" Testing with split value:", split_val)
            for i in range(epoch):
                if pre == 'tf_idf':
                    print("  Building RBF SVM with TF-IDF...")
                    svm_rbf(text_data, label_data, split_val)
                elif pre == 'bow':
                    print("  Building RBF SVM with BOW...")
                    svm_rbf_bow(text_data, label_data, split_val)
                run_val += 1
        print("Testing complete")
        rbf_file.close()

    if mod == 'poly':
        #run_val = 1
        print("Building and testing SVM with polynomial kernel...")
        for split_val in split_values:
            print(" Testing with split value:", split_val)
            for i in range(epoch):
                if pre == 'tf_idf':
                    print("  Building Poly SVM with TF-IDF...")
                    svm_poly(text_data, label_data, split_val)
                elif pre == 'bow':
                    print("  Building Poly SVM with BOW...")
                    svm_poly_bow(text_data, label_data, split_val)
                run_val += 1
        print("Testing complete")
        poly_file.close()

    # close the files at the end of execution


# END OF MAIN


#
# generate_SVM - takes training and testing splits for the dataset and builds
#       two SVM models: one using a polynomial kernel, the other with an RBF kernel.
#       The dataset is then tested over the constructed models, and accuracy is printed.
#
def generate_SVM(x_train, x_test, y_train, y_test):
    # create an SVM object: Using the Radial Basis Function
    rbf = svm.SVC(kernel='rbf', gamma=0.4, C=0.1).fit(x_train, y_train)

    # determine the efficiency
    rbf_pred = rbf.predict(x_test)

    rbf_accuracy = accuracy_score(y_test, rbf_pred)
    print("Accuracy of SVM with RBF kernel:", (rbf_accuracy*100))

    # create SVM object: Using the Polynomial function
    poly = svm.SVC(kernel='poly', degree=3, C=1).fit(x_train, y_train)

    poly_pred = poly.predict(x_test)

    poly_accuracy = accuracy_score(y_test, poly_pred)
    print("Accuracy of SVM with polynomial kernel:", (poly_accuracy*100))


#
# svm_linear - creates and tests an SVM model utilizing the linear kernel.
#           training and testing split is provided, and accuracy of model is printed.
#
def svm_linear(text_data, label_data, split):
    x_train, x_test, y_train, y_test = train_test_split(text_data, label_data, test_size=split)
    vectorize = TfidfVectorizer()
    train_vects = vectorize.fit_transform(x_train)
    test_vects = vectorize.transform(x_test)

    # create linear classifier
    svm_linear = svm.SVC(kernel='linear')
    svm_linear.fit(train_vects, y_train)
    pred_linear = svm_linear.predict(test_vects)
    linear_acc = accuracy_score(y_test, pred_linear)
    write_to_out(svm_linear_filename, linear_acc, run_val, split)


#
# svm_linear_bow - creates and tests an SVM model utilizing the linear kernel.
#       training and testing split provided. The data is pre-processed with
#       bag-of-words.
#
def svm_linear_bow(text_data, label_data, split):
    x_train, x_test, y_train, y_test = train_test_split(text_data, label_data, test_size=split)
    # create the count vectorizer
    count_vect = CountVectorizer()
    tf_train = count_vect.fit_transform(x_train)
    tf_test = count_vect.transform(x_test)

    # create linear classifier
    svm_linear = svm.SVC(kernel='linear')
    svm_linear.fit(tf_train, y_train)
    pred_linear = svm_linear.predict(tf_test)
    linear_acc = accuracy_score(y_test, pred_linear)
    write_to_out(svm_linear_filename, linear_acc, run_val, split)


#
# svm_rbf - creates and tests an SVM model utilizing the RBF kernel.
#          training and testing split is provided, and accuracy of model is printed.
#
def svm_rbf(text_data, label_data, split):
    x_train, x_test, y_train, y_test = train_test_split(text_data, label_data, test_size=split)
    vectorize = TfidfVectorizer()
    train_vects = vectorize.fit_transform(x_train)
    test_vects = vectorize.transform(x_test)

    # create rbf classifier
    svm_rbf = svm.SVC(kernel='rbf', gamma=0.4, C=0.1)
    svm_rbf.fit(train_vects, y_train)
    pred_rbf = svm_rbf.predict(test_vects)
    rbf_acc = accuracy_score(y_test, pred_rbf)
    write_to_out(svm_rbf_filename, rbf_acc, run_val, split)


#
# svm_rbf_bow - creates and tests an SVM model utilizing the RBF kernel.
#       training and testing split is provided. The data is pre-processed
#       with bag-of-words
#
def svm_rbf_bow(text_data, label_data, split):
    x_train, x_test, y_train, y_test = train_test_split(text_data, label_data, test_size=split)
    # create the count vectorizer
    count_vect = CountVectorizer()
    tf_train = count_vect.fit_transform(x_train)
    tf_test = count_vect.transform(x_test)

    # create the rbf classifier
    svm_rbf = svm.SVC(kernel='rbf', gamma=0.4, C=0.1)
    svm_rbf.fit(tf_train, y_train)
    pred_rbf = svm_rbf.predict(tf_test)
    rbf_acc = accuracy_score(y_test, pred_rbf)
    write_to_out(svm_rbf_filename, rbf_acc, run_val, split)


#
# svm_poly - creates and tests an SVM model utilizing the Polynomial kernel.
#       training and testing split is provided, and accuracy of the model is printed
#
def svm_poly(text_data, label_data, split):
    x_train, x_test, y_train, y_test = train_test_split(text_data, label_data, test_size=split)
    vectorize = TfidfVectorizer()
    train_vects = vectorize.fit_transform(x_train)
    test_vects = vectorize.transform(x_test)

    # create polynomial classifier
    svm_poly = svm.SVC(kernel='poly', degree=3, C=1)
    svm_poly.fit(train_vects, y_train)
    pred_poly = svm_poly.predict(test_vects)
    poly_acc = accuracy_score(y_test, pred_poly)
    write_to_out(svm_poly_filename, poly_acc, run_val, split)


#
# svm_poly_bow - creates and tests an SVM model utilizing the Polynomial kernel.
#       training and testing split is provided. The data is pre-processed 
#       with bag-of-words
#
def svm_poly_bow(text_data, label_data, split):
    x_train, x_test, y_train, y_test = train_test_split(text_data, label_data, test_size=split)
    # create the count vectorizer
    count_vect = CountVectorizer()
    tf_train = count_vect.fit_transform(x_train)
    tf_test = count_vect.transform(x_test)

    # create the polinomial classifier
    svm_poly = svm.SVC(kernel='poly', degree=3, C=1)
    svm_poly.fit(tf_train, y_train)
    pred_poly = svm_poly.predict(tf_test)
    poly_acc = accuracy_score(y_test, pred_poly)
    write_to_out(svm_poly_filename, poly_acc, run_val, split)


#
# write_to_out - takes the global filename and the accuracy rate and writes the output to the file.
#
def write_to_out(filename, acc_val, run, split_value):
    if (filename == svm_linear_filename):
        acc = str(run) + ":: " + str(split_value) + ": Accuracy Val: " + str(acc_val*100) + "\n"
        linear_file.write(acc)
    if (filename == svm_rbf_filename):
        acc = str(run) + ":: " + str(split_value) + ": Accuracy Val: " + str(acc_val*100) + "\n"
        rbf_file.write(acc)
    if (filename == svm_poly_filename):
        acc = str(run) + ":: " + str(split_value) + ": Accuracy Val:" + str(acc_val*100) + "\n"
        poly_file.write(acc)



if __name__ == "__main__":
    main()
