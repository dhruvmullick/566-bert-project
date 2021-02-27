import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.io


def get_words(proportion=70):
    proportion /= 100
    df = pd.read_csv('dataset.csv')
    words = list(df['Word'])
    for entry in range(len(words)): words[entry] = words[entry] + '$' + str(entry + 1)
    random.shuffle(words)
    partition_point = int(proportion * len(words))
    train_set, testing_set = words[:partition_point], words[partition_point:]
    # print(train_set)
    return [train_set, testing_set]
