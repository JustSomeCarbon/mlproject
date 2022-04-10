import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # read in the cleaned data
    dataset = pd.read_csv("cleandata.csv")
    text_data = dataset["Text"]
    label_data = dataset["Emotion"]


if __name__ == "__main__":
    main()
