"""
---------------------------------
Author: Zephyrus                |
Date: 06.22.2024                |
Purpose: Config File            |
---------------------------------
"""
import argparse

def get_options():
    parser = argparse.ArgumentParser(description='XGBOOST options')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for each update step')
    parser.add_argument('--optimizer', type=str, default='SGDW', help='Using GD for update')
    parser.add_argument('--max_depth', type=int, default=3, help='Maximum depth of the tree')
    parser.add_argument('--n_estimators', type=int, default=50, help='Maximum depth of the tree')
    parser.add_argument('--min_leaf', type=int, default=1, help='Minimum number of samples required to be at a leaf node')
    parser.add_argument('--gamma', type=float, default=0.1, help='Minimum loss reduction required to make a further partition')
    parser.add_argument('--lambda', type=float, default=1, help='L2 regularization term on weights')
    parser.add_argument('--min_child_weight', type=float, default=1, help='Minimum sum of instance weight (hessian) needed in a child')
    parser.add_argument('--eps', type=float, default=1e-5, help='A small number to prevent division by zero')
    parser.add_argument('--subsample_cols', type=float, default=1, help='Subsample ratio of the training instances')
    args = parser.parse_args()
    return args