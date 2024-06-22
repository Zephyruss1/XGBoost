"""
---------------------------------
Author: Zephyrus                |
Date: 06.22.2024                |
Purpose: Create Model           |
---------------------------------
"""
from sklearn.tree import DecisionTreeRegressor
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
        self.max_depth = args.max_depth
        self.n_estimators = args.n_estimators
        self.initial_prediction = self.max_depth
        self.trees = []
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
    
    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        prediction = np.full(y.shape, self.initial_prediction)

        for _ in range(self.n_estimators):
            gradients = y - prediction # pseudo-residuals

            # Fit a regression tree to the gradients
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, gradients)

            # Update the predictions
            prediction += self.lr * tree.predict(X)

            self.trees.append(tree)
            
    def predict(self, X):
        # Start with initial prediction
        prediction = np.full(X.shape[0], self.initial_prediction)
        
        # Add predictions from all trees
        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(X)
        
        return prediction



class Optimizations:
    def __init__(self, params):
        self.params = params