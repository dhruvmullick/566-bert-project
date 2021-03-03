from true_word_sourcer import *
from bert_embedding import *
from eeg_feature_extractor import *
from models.linear_regression import EegLinearRegression

train_set_sentences, testing_set_sentences = get_sentences_by_proportion()
eeg_representations = extract_eeg_feature_for_sentences(train_set_sentences)
eeg_representations_for_one_sentence_train = eeg_representations[0][0]
# print(min(len(eeg_representations_for_one_sentence_train),510))
eeg_representations_for_one_sentence_train = eeg_representations_for_one_sentence_train[:min(len(eeg_representations_for_one_sentence_train),510)]

bert_features_for_sentence_train = (get_bert_features_for_layer(train_set_sentences[0], 12))[1:min(510,len(eeg_representations_for_one_sentence_train))+1]

ridge_regression = EegLinearRegression()
score = ridge_regression.evaluate(eeg_representations_for_one_sentence_train, bert_features_for_sentence_train)
print(score)

# from src.true_word_sourcer import *
# from src.bert_embedding import *
# from src.eeg_feature_extractor import *
# from src.models.linear_regression import EegLinearRegression
#
#
# def train_and_test():
#     train_set_sentences, testing_set_sentences = get_sentences_by_proportion()
#
#     eeg_representations_train = extract_eeg_feature_for_sentences(train_set_sentences)
#     eeg_representations_train_averaged = average_over_participants(eeg_representations_train)
#
#     eeg_representations_test = extract_eeg_feature_for_sentences(testing_set_sentences)
#     eeg_representations_test_averaged = average_over_participants(eeg_representations_test)
#
#
#     bert_features_for_sentence_train = get_bert_features_for_layer(train_set_sentences[0], 12)
#     bert_features_for_sentence_test = get_bert_features_for_layer(train_set_sentences[1], 12)
#     ridge_regression = EegLinearRegression()
#     ridge_regression.evaluate([bert_features_for_sentence_train, bert_features_for_sentence_test],
#                               [eeg_representations_for_one_sentence_train, eeg_representations_for_one_sentence_test])
#
#
# def average_over_participants(eeg_representations):
#     return np.average(eeg_representations, axis=0)
#

