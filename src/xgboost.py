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
    def __init__(self, args, X_train, X_test, y_train, y_test, x, gradient, hessian, idxs):
        self.idxs = idxs
        self.x, self.gradient, self.hessian = x, gradient, hessian
        self.col_count = x.shape[1]
        self.row_count = len(idxs)
        self.column_subsample = np.random.permutation(self.col_count)[:round(self.subsample_cols*self.col_count)]
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
        self.min_leaf = args.min_leaf
        self.gamma = args.gamma
        self.lmbda = args.lmbda
        self.min_child_weight = args.min_child_weight
        self.eps = args.eps
        self.subsample_cols = args.subsample_cols
        # Initialize the prediction with the mean of the target values
        self.initial_prediction = np.mean(y_train)
        self.trees = []
        self.prediction = np.full(self.y_train.shape, self.initial_prediction)
    
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
        prediction = np.zeros(self.X_test.shape[0])
        
        # Add predictions from all trees
        for tree in self.trees:
            prediction += self.lr * tree.predict(self.X_test)
        
        return np.full((self.X_test.shape[0], 1), np.mean(self.y_test)).flatten() + prediction



class Optimizations:
    def __init__(self, params):
        self.params = params
