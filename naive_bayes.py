#
# file: naive_bayes.py
# description: This file contains the source code for the Naive Bayes model
#       implementation. The program expects the input to be already cleaned.
#

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
    print(bag_data.bag)

    return


if __name__ == '__main__':
    main()