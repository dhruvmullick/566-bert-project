from src.true_word_sourcer import *
from src.bert_embedding import *
from src.eeg_feature_extractor import *
from src.models import *

def train():
    words = get_words_for_training()
    eeg_representations = get_eeg_data_by_word(words)
    bert_features = get_bert_features(words)

