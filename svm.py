import math
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import seaborn as sns

nltk.download('punkt')

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

    text_term_freq = term_frequency(tok_frequency(text_data))
    frequency_count = word_per_doc(tok_frequency(text_data))
    idf = determine_idf(text_term_freq, frequency_count)

    for i in range(5):
        print(idf[i])


#
# tok_frequency - takes an array of input texts and returns the term frequency for each of the words
#           within each given text.
#
def tok_frequency(texts):
    total_freq = []

    for text in texts:
        toks = word_tokenize(text)
        local_freq = {}
        for t in toks:
            if t in local_freq:
                local_freq[t] += 1
            else:
                local_freq[t] = 1
        # append local word frequencies to total frequency array
        total_freq.append(local_freq)

    # return the total frequency array
    return total_freq


#
# term_frequency - takes a frequency matrix and determines the term frequency for each term
#           within a text.
#
def term_frequency(total_freq):
    tf_total = []

    for freq in total_freq:
        local_freq = {}
        text_len = len(freq)
        # determine the term frequency for each term
        for term, count in freq.items():
            local_freq[term] = count / text_len
        # append the local term frequency scores to array
        tf_total.append(local_freq)

    # return the term frequency array
    return tf_total


#
# word_per_doc - determines the total frequency of all terms within the texts.
#
def word_per_doc(total_freq):
    wpd = {}

    for freq in total_freq:
        for term, _ in freq.items():
            if term in wpd:
                wpd[term] += 1
            else:
                wpd[term] = 1

    return wpd


#
# determine_idf - calculates and returns an array containing a matrix of idf scores for
#           each term within a text, for all texts.
#
def determine_idf(term_freq, freq_count):
    idf = []
    total_texts = len(term_freq) # total number of

    for freq in term_freq:
        local_idf = {}
        for term, _ in freq.items():
            local_idf[term] = math.log10(total_texts / float(freq_count[term]))
        # append idf value to total idf array
        idf.append(local_idf)
    
    return idf


if __name__ == "__main__":
    main()
