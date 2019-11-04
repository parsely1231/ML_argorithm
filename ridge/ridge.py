import numpy as np
from scipy import linalg


class RidgeRegression(object):
    def __init__(self, lamb=1.):
        self.lamb = lamb
        self.w = None

    def fit(self, X, y):
        Xtil = np.hstack([np.ones(X.shape[0]).reshape(-1, 1), X])
        I = np.eye(Xtil.shape[1])
        A = np.dot(Xtil.T, Xtil) + self.lamb * I
        b = np.dot(Xtil.T, y)
        self.w = linalg.solve(A, b)

    def predict(self, X):
        if self.w is None:
            raise ValueError('No calculation model!! this class has not fitted.')
        Xtil = np.hstack([np.ones(X.shape[0]).reshape(-1, 1), X])
        return np.dot(Xtil, self.w)


