import math
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

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


#
# generate_tf_idf - generates and returns the tf-idf score for the given data.
#           expects a term-frequency array and a idf score array
#
def generate_tf_idf(term_freq, idf):
    tf_idf = []
    for t1, t2 in zip(term_freq, idf): # for each matrix within both arrays
        local_tf_idf = {}
        for (term, val1), (_, val2) in zip(t1.items(), t2.items()): # for each term and their values
            local_tf_idf[term] = float(val1 * val2)
        tf_idf.append(local_tf_idf)

    return tf_idf


#
# score_texts - takes the tf-idf scores and determines scores for each of the texts.
#
def score_texts(tf_idf):
    s_scores = []
    for texts in tf_idf:
        local_score = 0
        text_len = len(texts)
        for _, score in texts.items():
            local_score += score
        s_scores.append(local_score / text_len)

    return s_scores