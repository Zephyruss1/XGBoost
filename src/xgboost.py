"""
---------------------------------
Author: Zephyrus                |
Date: 06.22.2024                |
Purpose: Create Model           |
---------------------------------
"""
import xgboost as xgb
import optuna
from options import get_options

class XGBoost:
    def __init__(self):
        self.arg_params = get_options()
        self.lr = self.arg_params.lr
        self.optimizer = self.arg_params.optimizer
        self.iteration = self.arg_params.iteration
        self.gamma = self.arg_params.gamma
        self.model = None
        


    def fit(self, X, y):
        self.model = xgb.train(self.params, xgb.DMatrix(X, y))

    def predict(self, X):
        return self.model.predict(xgb.DMatrix(X))



class Optimizations:
    def __init__(self, params):
        self.params = params