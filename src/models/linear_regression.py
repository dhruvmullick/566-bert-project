from sklearn.linear_model import RidgeCV
import numpy as np

'''TODO: See how regularisation working. is it only over training set or not? 
After training, will we use the best regularisation or not. For MC. 
Check the training objective.
'''

class EegLinearRegression:

    def __init__(self, folds=5, regularisation_bound=(0.1, 100), max_iterations=100):
        '''need to see why logspace dependency not working'''
        alphas = np.logspace(regularisation_bound[0], regularisation_bound[1], num=500)
        # Uses K-Fold Regularisation here.
        self.model = RidgeCV(cv=folds, alphas=alphas, max_iter=max_iterations)

    def train(self, x, y):
        self.model.fit(x, y)

    # Change this to use Nested CV approach / (Monte Carlo + CV) approach for getting the alpha for from training set
    # and then making prediction for test set.
    def test(self, x):
        return self.model.predict(x)
