#
# file: naive_bayes.py
# description: This file contains the source code for the Naive Bayes model
#       implementation. The program expects the input to be already cleaned.
#

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB # multinomial naive bayes
import pandas as pd
import numpy as np
from bow import *

def main():
    dataset = pd.read_csv("cleandata.csv")

    # divide the dataset
    text_data = dataset["Text"]
    label_data = dataset["Emotion"]
    total_entries = len(text_data)

    # call for bag of words over data
    bag_data = Bow(text_data, label_data)
    #print(bag_data.bag)

    # create the Naive Bayes model
    nb_model(label_data, bag_data)


#
# nb_model - takes the label data from the dataset and a bag-of-words object
#       to create a multinomial Naive Bayes model
#
def nb_model(label_data, bag_data):
    # split the training and testing dataset
    x_train, x_test, y_train, y_test = train_test_split(bag_data.bag, label_data, test_size=0.2)
    
    # create the model
    mnd = MultinomialNB()
    mnd.fit(x_train, y_train)



if __name__ == '__main__':
    main()