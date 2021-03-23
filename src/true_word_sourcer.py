import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.io
import re


def get_sentence():
    with open('mod_sentence.txt') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = ''.join(content)
    content = content.lower()
    sentence = ""
    for character in range(len(content)):
        if content[character].isalpha() or content[character] == " " or content[character] == "?" or content[
            character] == "!" or content[character] == "." or (
                content[character] == "'" and content[character - 1] != " " and content[character + 1] != " "):
            sentence += content[character]

    sentence = sentence.replace('waistcoatpocket', 'waistcoat pocket')
    sentence = sentence.replace("'", ' ')

    # To include ? and ! apart from . and consider them as different sentence.
    sentence = sentence.replace("?", ".")
    sentence = sentence.replace("!", ".")

    sentence = sentence.split('.')
    print(len(sentence))
    return sentence


def get_sentences_by_proportion(proportion=80):
    proportion /= 100
    df = pd.read_csv('dataset.csv')
    words = list(df['Word'])
    for entry in range(len(words)):
        words[entry] = words[entry] + '$' + str(entry + 1)

    reference_sentence = get_sentence()  # These are simple sentences from dataset and would be used as an input to BERT
    sentences_for_EEG = []  # We will add an id to each word here. This will help us to find EEG representations sooner
    starting_index = 0
    for single_sentence in reference_sentence:
        reference_sentence_list = single_sentence.split(' ')
        idol_sentence_list = words[starting_index:starting_index + len(reference_sentence_list)]
        starting_index += len(reference_sentence_list)
        if len(' '.join(idol_sentence_list)): sentences_for_EEG.append(' '.join(idol_sentence_list))

    random.seed(4)
    random.shuffle(sentences_for_EEG)
    random.seed(4)
    random.shuffle(reference_sentence)
    # for i in sentences_for_EEG:print(i,'\n')
    # partition_point = int(proportion*len(sentences_for_EEG))
    # train_set,testing_set = sentences_for_EEG[:partition_point],sentences_for_EEG[partition_point:]
    return sentences_for_EEG, reference_sentence
