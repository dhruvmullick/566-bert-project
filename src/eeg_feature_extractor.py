import scipy
import numpy as np


def extract_eeg_feature_for_sentences(sentence_list):
    main_list = []
    for participant in range(1,24):   # change to number of participant
        if participant == 5:continue
        print('Participant : ',participant,' in progress')
        try:
          number = ''
          if participant<10:number=number+'0'+str(participant)
          else:number=number+str(participant)
          mat_file = scipy.io.loadmat('dataset/proc/S'+number+'.mat')    #Convert to loop by adding all the mat files
          #print('dataset/proc/S'+number+'.mat Found')
          raw_file = scipy.io.loadmat('dataset/EEG/S'+number+'.mat') #Convert to loop by adding all the dataset files.     
          #print('dataset/EEG/S'+number+'.mat Found')
          matrix = []
          for single_sentence in sentence_list:
              word_list = single_sentence.split(' ')
              row  =  []
              for single_word in word_list:
                  word,index = single_word.split('$')
                  index = int(index)
                  onset, offset = int(mat_file['proc']['trl'][0][0][index-1][0]),int(mat_file['proc']['trl'][0][0][index-1][1])
                  filtered_data = raw_file['raw']['trial'][0][0][0][0][:,onset-1:offset]
                  filtered_data = filtered_data.mean(axis = 1)
                  row.append(np.array(filtered_data))
              #print(len(row[0]))
              matrix.append(np.array(row))
                  
          main_list.append(np.array(matrix))
        except:
          print('Entry : ',participant,' Something Went Wrong')
          main_list.append(np.array([]))
    #print(len(main_list))
    return np.array(main_list)

#Participant - Sentence - Word
# data = eeg_feature_extractor(train_set)