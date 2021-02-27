from sklearn.linear_model import RidgeCV
import numpy as np

'''
Ridge Regression for applying on EEG data. 
Includes method abstractions for training, testing and an evaluator method call.  
'''


class EegLinearRegression:

    # Initialise the ridge regression model.
    def __init__(self, folds=5, regularisation_bound=(0.1, 200), max_iterations=500):
        # The regularisation parameters to select from.
        alphas = np.logspace(regularisation_bound[0], regularisation_bound[1], num=500)
        # Uses K-Fold CV here here. Does CV over alphas to get the best one.
        self.model = RidgeCV(cv=folds, alphas=alphas, max_iter=max_iterations, scoring="neg_mean_squared_error")

    # Fit the LR model on the input data.
    def train(self, x, y):
        self.model.fit(x, y)

    # Test the LR model to get the predicted score for the given input data.
    def test(self, x):
        return self.model.predict(x)

    # Evaluate the LR model by giving an accuracy value.
    def evaluation(self, x, y):
        pass
