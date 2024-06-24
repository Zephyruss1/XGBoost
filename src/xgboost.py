"""
---------------------------------
Author: Zephyrus                |
Date: 06.22.2024                |
Purpose: Create Model           |
---------------------------------
"""
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np
epsilon = 1e-5

class XGBoost:
    def __init__(self, args, X_train, X_test, y_train, y_test):
        # Initialize self variables
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self._weights = np.zeros_like(X_train)
        # Initialize argparse options
        self.lr = args.lr
        self.optimizer = args.optimizer
        self.max_depth = args.max_depth
        self.n_estimators = args.n_estimators
        # Initialize the prediction with the mean of the target values
        self.initial_prediction = np.mean(y_train)
        self.trees = []
        self.prediction = np.full(self.y_train.shape, self.initial_prediction)
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
    
    def fit_single_iteration(self):
        # Start with initial prediction
        prediction = np.full(self.y_train.shape, self.initial_prediction)
        for _ in range(self.n_estimators):
            gradients = self.y_train - prediction # pseudo-residuals
            # Fit a regression tree to the gradients
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(self.X_train, gradients)

            # Update the predictions
            self.prediction += self.lr * tree.predict(self.X_train)
            self.trees.append(tree)
        
    def mse(self):
        return np.mean((self.y_test - self.predict()) ** 2)
    
    def r2_score(self):
        y_mean = np.mean(self.y_test)
        sst = np.sum((self.y_test - y_mean) ** 2)
        ssr = np.sum((self.y_test - self.predict()) ** 2)
        r2 = 1 - (ssr / sst)
        return r2
    
    def predict(self):
        # Start with initial prediction
        prediction = np.full(self.X_test.shape[0], self.initial_prediction)
        
        # Add predictions from all trees
        for tree in self.trees:
            prediction += self.lr * tree.predict(self.X_test)
        
        return prediction



class Optimizations:
    def __init__(self, params):
        self.params = params
