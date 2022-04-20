import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import seaborn as sns
import tf_idf

from sklearn.feature_extraction.text import TfidfVectorizer

#
# Classes possible: Sadness, Anger, Love, Surprise, Fear, and Happiness (6)
#


def main():
    # read in the cleaned data
    dataset = pd.read_csv("cleandata.csv")


    # DIVIDE THE DATASET
    text_data = dataset["Text"]
    label_data = dataset["Emotion"]
    total_entries = len(text_data)
    print("Total text entries:", total_entries)

    print(text_data.shape, label_data.shape)

    # TF (term frequency) - how common a specific word is
    # TF(t) = (number of times term t appears in the document) / (total number of terms in the document)

    # IDF (inverse document frequency) - how unique or rare the word is
    # TDF(t) = log_e(total number of documents / number of documents with term t in it)

    # TF-IDF weight = TF * IDF  (product of the two quantities)

    text_term_freq = tf_idf.term_frequency(tf_idf.tok_frequency(text_data)) # frequency score of local terms : array (TF)
    frequency_count = tf_idf.word_per_doc(tf_idf.tok_frequency(text_data))  # frequency score of global terms : dictionary
    idf = tf_idf.determine_idf(text_term_freq, frequency_count)             # idf score determined from frequency : array (IDF)

    # generate the tf-idf product
    tf_idf_score = tf_idf.generate_tf_idf(text_term_freq, idf)  # TF-IDF score

    text_score = tf_idf.score_texts(tf_idf_score)               # Score array for each text in the data

    """
    for i in range(10):
        print(text_score[i])
    print("Average score:", tf_idf.average_score(text_score))
    """

    # training and testing split
    # x - random assortment of text_data
    # y - associated random assortment of label_data
    #x_train, x_test, y_train, y_test = train_test_split(text_score, label_data.to_list(), test_size=0.3)
    # access value : y_train[0]
    # access at index : y_train.index[0]

    # TESTING 123...
    x_train, x_test, y_train, y_test = train_test_split(text_data, label_data, test_size=0.3)
    test_Vectorize = TfidfVectorizer()
    test_train_vects = test_Vectorize.fit_transform(x_train)
    test_test_vects = test_Vectorize.transform(x_test)
    # create linear classifier
    test_svm_linear = svm.SVC(kernel='linear')
    test_svm_linear.fit(test_train_vects, y_train)
    test_pred_linear = test_svm_linear.predict(test_test_vects)
    test_linear_acc = accuracy_score(y_test, test_pred_linear)
    print("Accuracy of linear:", (test_linear_acc*100))


    #generate_SVM(x_train, x_test, y_train, y_test)


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
    print("Accuracy of SVM with RBF kernel: ", (rbf_accuracy*100))

    # create SVM object: Using the Polynomial function
    poly = svm.SVC(kernel='poly', degree=4, C=1).fit(x_train, y_train)

    poly_pred = poly.predict(x_test)

    poly_accuracy = accuracy_score(y_test, poly_pred)
    print("Accuracy of SVM with polynomial kernel: ", (poly_accuracy*100))




if __name__ == "__main__":
    main()
