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
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for each update step')
    parser.add_argument('--optimizer', type=str, default='SGDW', help='Using GD for update')
    parser.add_argument('--max_depth', type=int, default=3, help='Maximum depth of the tree')
    parser.add_argument('--n_estimators', type=int, default=100, help='Maximum depth of the tree')

    args = parser.parse_args()
    return args