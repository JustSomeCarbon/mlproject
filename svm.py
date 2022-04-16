import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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


#
# tok_frequency - takes an array of input texts and returns the term frequency for each of the words
#           within each given text.
#
def tok_frequency(texts):
    total_freq = []
    for text in texts:
        local_freq = {}
        for t in text:
            if t in local_freq:
                local_freq[t] += 1
            else:
                local_freq[t] = 1
        # append local word frequencies to total frequency array
        total_freq.append(local_freq)

    # return the total frequency array
    return term_frequency(total_freq)

#
# term_frequency - takes a frequency matrix and determines the term frequency for each term
#           within a given text.
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

if __name__ == "__main__":
    main()
