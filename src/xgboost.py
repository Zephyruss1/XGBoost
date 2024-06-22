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
epsilon = 1e-5

class XGBoost:
    def __init__(self, args, X_train, X_test, y_train, y_test):
        # Load data
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self._weights = np.zeros_like(X_train)
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
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, value):
        assert isinstance(value, np.ndarray) and value.shape == self._weights.shape, "weights must be a numpy array and have the same shape as the current weights"
        self._weights = value
    
    def fit(self):
        self.model = xgb.train({
            'learning_rate': self.lr,
            'gamma': self.gamma,
            'objective': 'reg:squarederror'
        }, xgb.DMatrix(self.X_train, self.y_train), num_boost_round=self.iteration)
        
    def predict(self):
        return self.model.predict(xgb.DMatrix(self.X_test))



class Optimizations:
    def __init__(self, params):
        self.params = params