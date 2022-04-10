import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# download stopwords
nltk.download("stopwords")
nltk.download('punkt')


def main():
    # read the dataset
    dataset = pd.read_csv("dataset/sentiment1/Emotion_final.csv")
    labels = dataset["Emotion"].unique() # sadness, anger, love, surprise, fear, happy
    dataset = dataset.dropna()

    # store input data and labels
    in_data = dataset.drop('Emotion', axis=1)
    label_data = dataset["Emotion"]

    # tokenize strings
    new_data = []
    for s in in_data["Text"]:
        tok = word_tokenize(s.lower())
        new_data.append(tok)

    # stop word construction
    stop_words = set(stopwords.words('english'))

    # remove the stopwords from each input data
    for i, n in enumerate(new_data):
        # temprary store for current string array
        tmp = []
        for str in n:
            if str not in stop_words:
                tmp.append(str)
        # set the data without stopwords
        new_data[i] = tmp

    print(new_data[0])


if __name__ == "__main__":
    main()
