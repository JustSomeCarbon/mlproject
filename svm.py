import math
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import seaborn as sns
import tf_idf

def main():
    # read in the cleaned data
    dataset = pd.read_csv("cleandata.csv")

    # DIVIDE THE DATASET
    text_data = dataset["Text"]
    label_data = dataset["Emotion"]
    total_entries = len(text_data)


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

# END OF MAIN


if __name__ == "__main__":
    main()
