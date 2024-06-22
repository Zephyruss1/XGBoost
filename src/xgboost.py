"""
---------------------------------
Author: Zephyrus                |
Date: 06.22.2024                |
Purpose: Create Model           |
---------------------------------
"""
import xgboost as xgb
import optuna
import numpy as np

class XGBoost:
    def __init__(self, args, X_train, X_test, y_train, y_test):
        # Load data
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Load options
        self.lr = args.lr
        self.optimizer = args.optimizer
        self.iteration = args.iteration
        self.gamma = args.gamma
        # Initialize M & N for specific optimizers
        self.m = np.zeros_like(self.weights)
        self.n = np.zeros_like(self.weights)
        self.t = 0

    @property
    def _weights(self):
        return self.weights
    
    @_weights.setter
    def _weights(self, value):
        assert isinstance(value, np.ndarray) and value.shape == self._weights.shape, "weights must be a numpy array and have the same shape as the current weights"
        self.weights = value
        
    def fit(self, X, y):
        self.model = xgb.train(self.params, xgb.DMatrix(X, y))

    def predict(self, X):
        return self.model.predict(xgb.DMatrix(X))



class Optimizations:
    def __init__(self, params):
        self.params = params