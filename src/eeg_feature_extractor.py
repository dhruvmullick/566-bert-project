def get_eeg_data_for_words(word_list):
    main_list = []
    for participant in range(1):   # change to number of participant
        single_participant = {}
        mat_file = scipy.io.loadmat('proc/S01.mat')    #Convert to loop by adding all the mat files
        raw_file = scipy.io.loadmat('S01_dataset.mat') #Convert to loop by adding all the dataset files.     
        #print(raw_file['raw']['trial'][0][0][0][0].shape)
        for single_word in word_list:
            word,index = single_word.split('$')
            index = int(index)
            onset, offset = int(mat_file['proc']['trl'][0][0][index-1][0]),int(mat_file['proc']['trl'][0][0][index-1][1])
            filtered_data = raw_file['raw']['trial'][0][0][0][0][:,onset:offset+1]
            try:
                single_participant[word].append(filtered_data)
            except:
                single_participant[word]=[filtered_data]
        main_list.append(single_participant)
    return main_list