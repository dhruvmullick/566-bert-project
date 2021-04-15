from true_word_sourcer import *
from bert_embedding import *
from eeg_feature_extractor import *
from data_processor import *
from models.linear_regression import EegLinearRegression
from models.neural_network import NeuralNetwork
from sklearn import preprocessing
from sklearn.decomposition import PCA
import json

MAX_BERT_SIZE = 512 - 2
layer_numbers = [12]

ENABLE_SLIDING_WINDOW = False  # if true, dim(EEG word represenations) = 372, o/w 62
ENABLE_BERT_LAYER_CONCAT = False
ENABLE_DIM_RED = False  # perform dimensionality reduction step
ENABLE_STATISTICAL_TEST = False


def create_PCA_representation(X, scale=True, var_cutoff=0.8):
    if scale:  # https://stats.stackexchange.com/questions/69157/why-do-we-need-to-normalize-data-before-principal-component-analysis-pca
        std_slc = preprocessing.StandardScaler()
        X = std_slc.fit_transform(X)

    pca = PCA(n_components=var_cutoff)
    X_pca = pca.fit_transform(X)

    print("Before PCA:", X.shape,
          "\nAfter PCA:", X_pca.shape,
          "\nPOVE:", sum(pca.explained_variance_ratio_))

    return X_pca


sentences_for_eeg, sentences_for_BERT = get_sentences_by_proportion()

# create EEG representations based on if we want to use the sliding window technique or not
if ENABLE_SLIDING_WINDOW:
    eeg_representations = extract_eeg_window_feature_for_sentences(sentences_for_eeg)
else:
    eeg_representations = extract_eeg_feature_for_sentences(sentences_for_eeg)

eeg_representations_averaged = average_eeg_over_participants(eeg_representations)
eeg_representations_truncated = []

for i in range(eeg_representations_averaged.shape[0]):
    eeg_representations_truncated.append(
        eeg_representations_averaged[i][:min(len(eeg_representations_averaged[i]), MAX_BERT_SIZE)])
eeg_representations_truncated = np.array(eeg_representations_truncated)

bert_features_for_sentences = []

for i in range(len(sentences_for_eeg)):
    features_list = []  # list of layers we're considering
    for layer_number in layer_numbers:
        features = get_bert_features_for_layer(sentences_for_BERT[i], layer_number)  # (512, 768)
        features = features[
                   1:min(MAX_BERT_SIZE, eeg_representations_truncated[i].shape[0]) + 1]  # (num words in sentence, 768)
        features_list.append(features)
    if ENABLE_BERT_LAYER_CONCAT:  # concat BERT layers
        features = np.concatenate(features_list, axis=1)  # (num words in sentence, 3072)
    else:  # average BERT layers
        features = sum(features_list) / len(layer_numbers)  # (num words in sentence, 762)
    print(features.shape)
    bert_features_for_sentences.append(features)
bert_features_for_sentences = np.array(bert_features_for_sentences)

ridge_regression = EegLinearRegression()
neural_network = NeuralNetwork(output_size=60)
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

if ENABLE_DIM_RED:
    flatten_bert_features_for_sentences = create_PCA_representation(flatten_bert_features_for_sentences)

# SEEING NAN VALUES FOR THIS WORD. SO JUST REMOVE THEM.
flatten_eeg_representations_truncated = np.delete(flatten_eeg_representations_truncated, (862), axis=0)
flatten_bert_features_for_sentences = np.delete(flatten_bert_features_for_sentences, (862), axis=0)

# test to see if any NaNs or Infs in data
if np.isnan(flatten_bert_features_for_sentences).any().any() or np.isinf(
        flatten_bert_features_for_sentences).any().any():
    print("badness in BERT")
    exit()
if np.isnan(flatten_eeg_representations_truncated).any().any() or np.isinf(
        flatten_eeg_representations_truncated).any().any():
    print("badness in EEG")
if np.any(np.isnan(flatten_eeg_representations_truncated)) or np.any(np.isnan(flatten_bert_features_for_sentences)):
    print("issue with nan")
if not np.all(np.isfinite(flatten_eeg_representations_truncated)) or not np.all(
        np.isfinite(flatten_bert_features_for_sentences)):
    print("issue with finite")

if ENABLE_STATISTICAL_TEST:
    file1 = open("permutation_results.txt", "w")
    for test_count in range(0, 20):
        np.random.shuffle(flatten_eeg_representations_truncated)
        score = ridge_regression.evaluate(flatten_eeg_representations_truncated, flatten_bert_features_for_sentences)
        file1.write(str(score))
        print(score)
    file1.close()

# score,weights = ridge_regression.evaluate(flatten_bert_features_for_sentences,flatten_eeg_representations_truncated)
score, weights = ridge_regression.evaluate(flatten_eeg_representations_truncated, flatten_bert_features_for_sentences)
weights = weights[0]

random_1 = weights[150]
random_2 = weights[550]
random_3 = weights[680]

with open("random1.txt", "w") as output:
    output.write(str(random_1))

with open("random2.txt", "w") as output:
    output.write(str(random_2))

with open("random3.txt", "w") as output:
    output.write(str(random_3))

# weights = str(weights)
# text_file = open("optimal_weights_eeg_bert.txt", "w")
# text_file.write(weights)
# text_file.close()
# score = neural_network.evaluate(flatten_eeg_representations_truncated,flatten_bert_features_for_sentences)
# score = neural_network.evaluate(flatten_bert_features_for_sentences,flatten_eeg_representations_truncated)

print(score)
print(weights)
