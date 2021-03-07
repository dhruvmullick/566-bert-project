from true_word_sourcer import *
from bert_embedding import *
from eeg_feature_extractor import *
from data_processor import *
from models.linear_regression import EegLinearRegression

MAX_BERT_SIZE = 512 - 2

train_set_sentences, testing_set_sentences = get_sentences_by_proportion()
sentences = train_set_sentences + testing_set_sentences

eeg_representations = extract_eeg_feature_for_sentences(sentences)
eeg_representations_averaged = average_eeg_over_participants(eeg_representations)
eeg_representations_truncated = np.zeros(eeg_representations_averaged.shape)

for i in range(eeg_representations_averaged.shape[0]):
    eeg_representations_truncated[i] = eeg_representations_averaged[i][:min(len(eeg_representations_averaged[i]), MAX_BERT_SIZE)]

bert_features_for_sentences = np.array(len(sentences))
for i in range(len(sentences)):
    features = get_bert_features_for_layer(sentences[i], 12)
    features = features[1:min(MAX_BERT_SIZE, eeg_representations_truncated[i].shape[0]) + 1]
    bert_features_for_sentences[i] = features

ridge_regression = EegLinearRegression()
score = ridge_regression.evaluate(eeg_representations_truncated.ravel(), bert_features_for_sentences.ravel())
print(score)
