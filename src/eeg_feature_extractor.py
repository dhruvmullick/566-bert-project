import scipy
import numpy as np
def extract_eeg_feature_for_sentences(sentence_list):
    main_list = []
    for participant in range(1):   # change to number of participant
        #single_participant = {}
        mat_file = scipy.io.loadmat('S01.mat')    #Convert to loop by adding all the mat files
        raw_file = scipy.io.loadmat('S01_dataset.mat') #Convert to loop by adding all the dataset files.     
        #print(raw_file['raw']['trial'][0][0][0][0].shape)
        matrix = []
        for single_sentence in sentence_list:
            word_list = single_sentence.split(' ')
            row  =  []
            for single_word in word_list:
                word,index = single_word.split('$')
                index = int(index)
                onset, offset = int(mat_file['proc']['trl'][0][0][index-1][0]),int(mat_file['proc']['trl'][0][0][index-1][1])
                filtered_data = raw_file['raw']['trial'][0][0][0][0][:,onset-1:offset]
#                 try:
#                     single_participant[word].append(filtered_data)
#                 except:
#                     single_participant[word]=[filtered_data]
                
                filtered_data = filtered_data.mean(axis = 1)
                row.append(filtered_data)
            matrix.append(row)
                
        #main_list.append(single_participant)
        main_list.append(matrix)
    return main_list

#Participant - Sentence - Word
# data = eeg_feature_extractor(train_set)