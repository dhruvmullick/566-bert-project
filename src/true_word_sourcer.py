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
        if content[character].isalpha() or content[character] == " " or content[character] == "." or (content[character] == "'" and content[character-1]!=" " and content[character+1]!=" "):
            sentence+=content[character]


    sentence = sentence.replace('waistcoatpocket','waistcoat pocket')
    sentence = sentence.replace("'",' ')
    sentence = sentence.split('.')
    return sentence

    
def get_sentences_by_proportion(proportion=80):
    proportion/=100
    df = pd.read_csv('dataset.csv')
    words = list(df['Word'])
    for entry in range(len(words)):words[entry] = words[entry]+'$'+str(entry+1)
    reference_sentence = get_sentence()
    final_sentence = []
    starting_index = 0
    for single_sentence in reference_sentence:
        reference_sentence_list = single_sentence.split(' ')
        idol_sentence_list = words[starting_index:starting_index+len(reference_sentence_list)]
        starting_index += len(reference_sentence_list)
        if len(' '.join(idol_sentence_list)):final_sentence.append(' '.join(idol_sentence_list))
    random.shuffle(final_sentence)
    for i in final_sentence:print(i,'\n')
    partition_point = int(proportion*len(final_sentence))
    train_set,testing_set = final_sentence[:partition_point],final_sentence[partition_point:]
    return train_set, testing_set

# data = true_word_sourcer(90)
# train_set = data[0]
# test_set = data[1]
