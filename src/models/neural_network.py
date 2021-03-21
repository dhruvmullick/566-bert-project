from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
import numpy as np

'''
Neural Networks for applying on EEG data. 
To be accessed through evaluate() method call. This will abstract the training, testing process as per the evaluation method selected.  
'''


class NeuralNetwork:
    # Initialise the ridge regression model.
    def __init__(self, folds=5, num_epochs=10, batch_size=100, hidden_layer_size_range=range(16, 64, 16)):
        # Uses K-Fold CV here here. Does CV over alphas to get the best one.
        self.folds = folds
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.hidden_layer_size_range = hidden_layer_size_range

    # Define the NN model to use for a given number of hidden layer size.
    def __nn_model(self, num_hidden):
        model = Sequential()
        model.add(Dense(num_hidden, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    # Fit the NN model on the input data.
    def __train(self, model, x, y):
        return model.fit(x, y, batch_size=self.batch_size, epochs=self.num_epochs, verbose=0)

    # Test the NN model to get the predicted score for the given input data.
    def __test(self, model, x, y):
        return model.evaluate(x, y, verbose=0)

    # Calculate the nested cv score.
    def __nested_cv_score(self, x, y, inner_folds=5, outer_folds=5):
        inner_kfold = KFold(n_splits=inner_folds, shuffle=True)
        outer_kfold = KFold(n_splits=outer_folds, shuffle=True)
        outer_cv_scores = []
        for outer_train_idx, outer_test_idx in outer_kfold.split(x):
            least_hidden_layer_size = 0
            least_score = 10000000000000000
            for hidden_layer_size in self.hidden_layer_size_range:
                acc_per_fold = []
                for inner_train_idx, inner_test_idx in inner_kfold.split(outer_train_idx):
                    # Define the model
                    model = self.__nn_model(hidden_layer_size)
                    # Fit data to model
                    history = self.__train(model, x[outer_train_idx[inner_train_idx]], y[outer_train_idx[inner_train_idx]])
                    # Generate generalization metrics
                    score = self.__test(model, x[outer_train_idx[inner_test_idx]], y[outer_train_idx[inner_test_idx]])
                    acc_per_fold.append(score)
                avg_accuracy_for_hidden_layer = np.mean(acc_per_fold)
                if avg_accuracy_for_hidden_layer < least_score:
                    least_score = avg_accuracy_for_hidden_layer
                    least_hidden_layer_size = hidden_layer_size

            print(least_score)

            model = self.__nn_model(least_hidden_layer_size)
            history = self.__train(model, x[outer_train_idx], y[outer_train_idx])
            # Generate generalization metrics
            outer_score = self.__test(model, x[outer_test_idx], y[outer_test_idx])
            outer_cv_scores.append(outer_score)
        return np.mean(outer_cv_scores)

    # Evaluate the model by giving an accuracy value. Will accept type of evaluation as well.
    def evaluate(self, x, y):
        return self.__nested_cv_score(x, y)
