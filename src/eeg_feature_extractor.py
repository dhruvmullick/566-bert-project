import scipy
import numpy as np
import sklearn
from true_word_sourcer import *
from bert_embedding import *


def extract_eeg_feature_for_sentences(sentence_list):
    main_list = []
    not_present = [47, 46, 33, 31, 29, 28, 38]
    participant_filter = [4, 23, 25, 26, 27, 28, 29, 30, 31]
    participant_counter = -1
    # red_word = [36,35,34,32,30,24,26,27]

    ''' CHANGING THE PARTI. NO. TO 15, CAN BE CHANGED TO 42 LATER '''

    for participant in range(1, 42):  # change to number of participant
        if participant in not_present: continue
        participant_counter += 1
        if participant_counter in participant_filter: continue

        print('Participant : ', participant, ' in progress')
        number = ''
        if participant < 10:
            number = number + '0' + str(participant)
        else:
            number = number + str(participant)
        mat_file = scipy.io.loadmat('dataset/proc/S' + number + '.mat')  # Convert to loop by adding all the mat files
        # print('dataset/proc/S'+number+'.mat Found')
        raw_file = scipy.io.loadmat(
            'dataset/EEG/S' + number + '.mat')  # Convert to loop by adding all the dataset files.
        # print('dataset/EEG/S'+number+'.mat Found')
        matrix = []
        for single_sentence in sentence_list:
            word_list = single_sentence.split(' ')
            row = []
            flag = 0  # to check which sentence is read properly
            counter = 1
            for single_word in word_list:
                word, index = single_word.split('$')
                index = int(index)
                try:
                    onset, offset = int(mat_file['proc']['trl'][0][0][index - 1][0]), int(
                        mat_file['proc']['trl'][0][0][index - 1][1])
                    filtered_data = raw_file['raw']['trial'][0][0][0][0][:, onset - 1:offset]
                    filtered_data = filtered_data.mean(axis=1)
                    row.append(np.array(filtered_data[:60]))
                    flag = counter
                    counter += 1
                except:
                    # print(word,index)
                    counter += 1
                    continue
            # print(participant,len(row[0]),len(row))
            # print(len(row))
            matrix.append(np.array(row))
        if flag != 0:
            print(len(matrix[flag][0]), len(matrix))
        else:
            print("Vocabulary Issue")
        main_list.append(np.array(matrix))

    # print(len(main_list))
    return np.array(main_list)


def extract_eeg_window_feature_for_sentences(sentence_list):
    main_list = []
    not_present = [47, 46, 33, 31, 29, 28, 38]
    participant_filter = [4, 23, 25, 26, 27, 28, 29, 30, 31]
    participant_counter = -1
    # red_word = [36,35,34,32,30,24,26,27]

    for participant in range(1, 42):  # change to number of participant
        if participant in not_present: continue
        participant_counter += 1
        if participant_counter in participant_filter: continue

        print('Participant : ', participant, ' in progress')
        try:
            number = ''
            if participant < 10:
                number = number + '0' + str(participant)
            else:
                number = number + str(participant)
            mat_file = scipy.io.loadmat(
                'dataset/proc/S' + number + '.mat')  # Convert to loop by adding all the mat files
            # print('dataset/proc/S'+number+'.mat Found')
            raw_file = scipy.io.loadmat(
                'dataset/EEG/S' + number + '.mat')  # Convert to loop by adding all the dataset files.
            # print('dataset/EEG/S'+number+'.mat Found')
            matrix = []
            for single_sentence in sentence_list:
                word_list = single_sentence.split(' ')
                row = []
                flag = 0  # to check which sentence is read properly
                counter = 1
                for single_word in word_list:
                    word, index = single_word.split('$')
                    index = int(index)
                    try:
                        onset, offset = int(mat_file['proc']['trl'][0][0][index - 1][0]), int(
                            mat_file['proc']['trl'][0][0][index - 1][1])
                        filtered_data = raw_file['raw']['trial'][0][0][0][0][:,
                                        onset - 1:offset]  # [62,651]
                        concat_vec = []
                        for item in [275, 300, 325, 350, 375,
                                     400]:  # Decoding word and category-specific spatiotemporal representations from MEG and EEG
                            concat_vec.extend(filtered_data[:, item:item + 25].mean(axis=1))
                        row.append(np.array(concat_vec))
                        flag = counter
                        counter += 1
                    except:
                        counter += 1
                        continue

                    # if np.isnan(np.array(concat_vec)).any() or np.isinf(np.array(concat_vec)).any():
                    #   print("badness")
                # print(len(row[0]))
                matrix.append(np.array(row))

            if flag != 0:
                print(len(matrix[flag][0]), len(matrix))
            else:
                print("Vocabulary Issue")
            main_list.append(np.array(matrix))
        except:
            print('Entry : ', participant, ' Something Went Wrong')
            main_list.append(np.array([]))
    # print(len(main_list))
    return np.array(main_list)

# Participant - Sentence - Word
# data = eeg_feature_extractor(train_set)


# sentences_for_eeg, sentences_for_BERT = get_sentences_by_proportion()
# eeg_representations = extract_eeg_feature_for_sentences(sentences_for_eeg)
