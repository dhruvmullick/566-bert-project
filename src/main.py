from src.true_word_sourcer import *
from src.bert_embedding import *
from src.eeg_feature_extractor import *
from src.models.linear_regression import EegLinearRegression

def train():
    words = get_words()
    eeg_representations = get_eeg_data_for_words(words)
    bert_features = get_bert_features(words)
    ridge_regression = EegLinearRegression()


