"""
---------------------------------
Author: Zephyrus                |
Date: 06.22.2024                |
Purpose: Create Model           |
---------------------------------
"""
import xgboost as xgb
import optuna

class XGBoost:
    def __init__(self, params):
        self.params = params
        self.model = None


    def fit(self, X, y):
        self.model = xgb.train(self.params, xgb.DMatrix(X, y))

    def predict(self, X):
        return self.model.predict(xgb.DMatrix(X))



class Optimizations:
    def __init__(self, params):
        self.params = params