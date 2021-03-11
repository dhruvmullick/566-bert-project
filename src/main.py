from true_word_sourcer import *
from bert_embedding import *
from eeg_feature_extractor import *
from data_processor import *
from models.linear_regression import EegLinearRegression
from sklearn import preprocessing

MAX_BERT_SIZE = 512 - 2
layer_numbers = [9]

train_set_sentences, testing_set_sentences = get_sentences_by_proportion()
sentences = train_set_sentences + testing_set_sentences



eeg_representations = extract_eeg_feature_for_sentences(sentences)
eeg_representations_averaged = average_eeg_over_participants(eeg_representations)
eeg_representations_truncated = []

for i in range(eeg_representations_averaged.shape[0]):
    eeg_representations_truncated.append(eeg_representations_averaged[i][:min(len(eeg_representations_averaged[i]), MAX_BERT_SIZE)])
eeg_representations_truncated = np.array(eeg_representations_truncated)



bert_features_for_sentences = []

for i in range(len(sentences)):
    features_list = []
    for layer_number in layer_numbers:
      features = get_bert_features_for_layer(sentences[i], layer_number)
      features_list.append(features)
    features = sum(features_list)/len(layer_numbers)

    features = features[1:min(MAX_BERT_SIZE, eeg_representations_truncated[i].shape[0]) + 1]
    bert_features_for_sentences.append(features)
bert_features_for_sentences = np.array(bert_features_for_sentences)
ridge_regression = EegLinearRegression()

flatten_eeg_representations_truncated = []
for i in range(len(eeg_representations_truncated)):
  flatten_eeg_representations_truncated.extend(eeg_representations_truncated[i])

min_max_scaler = preprocessing.MinMaxScaler()
flatten_eeg_representations_truncated = min_max_scaler.fit_transform(flatten_eeg_representations_truncated)
flatten_bert_features_for_sentences = []
for i in range(len(bert_features_for_sentences)):
  flatten_bert_features_for_sentences.extend(bert_features_for_sentences[i])

flatten_eeg_representations_truncated = np.array(flatten_eeg_representations_truncated)
flatten_bert_features_for_sentences = np.array(flatten_bert_features_for_sentences)

score = ridge_regression.evaluate(flatten_eeg_representations_truncated, flatten_bert_features_for_sentences)
print(score)
