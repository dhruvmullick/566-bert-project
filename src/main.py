from src.true_word_sourcer import *
from src.bert_embedding import *
from src.eeg_feature_extractor import *
from src.models.linear_regression import EegLinearRegression

train_set_sentences, testing_set_sentences = get_sentences_by_proportion()
eeg_representations = extract_eeg_feature_for_sentences(train_set_sentences)
eeg_representations_for_one_sentence_train = eeg_representations[0][0]
eeg_representations_for_one_sentence_test = eeg_representations[0][1]
bert_features_for_sentence_train = get_bert_features_for_layer(train_set_sentences[0], 12)
bert_features_for_sentence_test = get_bert_features_for_layer(train_set_sentences[1], 12)
ridge_regression = EegLinearRegression()
ridge_regression.evaluate([bert_features_for_sentence_train, bert_features_for_sentence_test],
                          [eeg_representations_for_one_sentence_train, eeg_representations_for_one_sentence_test])
