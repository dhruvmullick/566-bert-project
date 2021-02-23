from sklearn.linear_model import LogisticRegressionCV
import numpy as np

'''TODO: See how regularisation working. is it only over training set or not? 
After training, will we use the best regularisation or not. For MC. 
Check the training objective.
'''

class EegLogisticRegression:

    def __init__(self, folds=5, regularisation_bound=(0.1, 100), max_iterations=100):
        '''need to see why logspace dependency not working'''
        c_values = np.logspace(1 / regularisation_bound[1], 1 / regularisation_bound[0], num=500)
        self.model = LogisticRegressionCV(cv=folds, penalty='l2', Cs=c_values, max_iter=max_iterations)

    def train(self, x, y):
        self.model.fit(x, y)

    def test(self, x):
        return self.model.predict(x)
