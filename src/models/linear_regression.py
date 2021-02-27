import enum

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

'''
Ridge Regression for applying on EEG data. 
To be accessed through evaluation() method call. This will abstract the training, testing process as per the evaluation method selected.  
'''


class EegLinearRegression:
    cv_types = enum.Enum('nested_cv')

    # Initialise the ridge regression model.
    def __init__(self, folds=5, regularisation_bound=(0.1, 200), max_iterations=500):
        # The regularisation parameters to select from.
        alphas = np.logspace(regularisation_bound[0], regularisation_bound[1], num=500)
        # Uses K-Fold CV here here. Does CV over alphas to get the best one.
        self.model = RidgeCV(cv=folds, alphas=alphas, max_iter=max_iterations, scoring="neg_mean_squared_error")

    # Fit the LR model on the input data.
    def __train(self, x, y):
        return self.model.fit(x, y)

    # Test the LR model to get the predicted score for the given input data.
    def __test(self, x, y):
        return self.model.score(x, y)

    # Calculate the nested cv score.
    def __nested_cv_score(self, x, y, outer_folds=5):
        outer_cv = KFold(outer_folds, shuffle=True)
        scores = []
        for train_data_ind, test_data_ind in outer_cv.split(x, y):
            x_train, x_test = x[train_data_ind, :], x[test_data_ind, :]
            y_train, y_test = y[train_data_ind], y[test_data_ind]
            # Trains RidgeCV with cross validation.
            self.__train(x_train, y_train)
            best_score_for_fold = self.__test(x_test, y_test)
            scores += [best_score_for_fold]
        return np.average(np.array(scores))

    # Evaluate the LR model by giving an accuracy value. Will accept type of evaluation as well.
    def evaluation(self, x, y, cv_type=cv_types.nested_cv):
        if cv_type == self.cv_types.nested_cv:
            return self.__nested_cv_score(self, x, y)
