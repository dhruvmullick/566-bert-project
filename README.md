## 566-bert-project : Are Multi-Layer BERT Features and EEG Data Correlated?
![](images/updated_overview.png)

### Table of Contents

- [Description](#description)
- [How To Use](#how-to-use)
- [Code Details](#code-details)


## Description
Literature in NLP suggests that we can get better NLP task results by using features from multiple layers ([Devlin et al., 2019](https://www.aclweb.org/anthology/N19-1423/)), and we know that alignment of models with brain data leads to an increased performance on tasks ([Hollenstein et al., 2019a](https://arxiv.org/abs/1904.02682); [Toneva and Wehbe, 2019](https://arxiv.org/abs/1905.11833)), we hypothesise that the multi-layer features are more correlated with the brain data than single layer features. If we know which multi-layer features are most cor- related with brain data, then we can use them for getting improved results on NLP tasks. 

With the progress seen in NLP by using the recent BERT model, and the improvement in performance seen by applying brain data to models ([Hollenstein et al., 2020](https://www.aclweb.org/anthology/2020.lincr-1.3/)), we feel encouraged to pursue this problem of exploring correlations between BERT representations and brain activity for semantically and/or syntactically incorrect data.

We test this hypothesis empirically in our project.

For more details, please check the [video](https://www.youtube.com/watch?v=zb4UGBLtmpo) uploaded. 

## How To Use

#### Installation

* Install required packages using `pip install -r requirements.txt`
* Run the main.py : `python main.py`

### Experiments

* To enable the sliding window in the codebase, change the value of Boolean variable to True in main.py
```html
ENABLE_SLIDING_WINDOW = False
```
* To enable the BERT layer concatenation in the codebase, change the value of ENABLE_BERT_LAYER_CONCAT variable to True in main.py
```html
ENABLE_BERT_LAYER_CONCAT = False
```
* To enable the dimensionality reduction in the codebase, change the value of ENABLE_DIM_RED variable to True in main.py
```html
ENABLE_DIM_RED = False
```
* To enable the statistical test in the code in the codebase, change the value of ENABLE_STATISTICAL_TEST variable to True in main.py
```html
ENABLE_STATISTICAL_TEST = False
``` 

## Code Details

We have the following modules in our code: 
* [main.py](https://github.com/dhruvmullick/566-bert-project/blob/main/src/main.py): The entry point for our project where we orchestrate the different modules in our code and also set variables to be used in experiments (as mentioned above) 
* [linear_regression.py](https://github.com/dhruvmullick/566-bert-project/blob/main/src/models/linear_regression.py): Holds the model to be used for implementing a nested cross validation evaluation technique over linear regression.
* [neural_network.py](https://github.com/dhruvmullick/566-bert-project/blob/main/src/models/neural_network.py): Holds the model to be used for implementing a nested cross validation evaluation technique over 1 hidden layer neural network.
* [bert_embedding.py](https://github.com/dhruvmullick/566-bert-project/blob/main/src/bert_embedding.py): extracts embeddings for the sentence tokens from BERT for the required layers. 
* [data_processor.py](https://github.com/dhruvmullick/566-bert-project/blob/main/src/data_processor.py): Some utility methods. 
* [eeg_feature_extractor.py](https://github.com/dhruvmullick/566-bert-project/blob/main/src/eeg_feature_extractor.py): Extract EEG features for the different words from Alice dataset. 
* [plot_statistical_tests.py](https://github.com/dhruvmullick/566-bert-project/blob/main/src/plot_statistical_tests.py): Plot the histograms for the saved data file for permutation tests (which were genereated from main.py). 
* [true_word_sourcer.py](https://github.com/dhruvmullick/566-bert-project/blob/main/src/true_word_sourcer.py): Get the words used in Alice dataset. 


[Back To The Top](#566-bert-project )
