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
    parser.add_argument('--lr', tpye=float, default=3e-4, help='Learning rate for each update step')
    parser.add_argument('--optimizer', type=str, default='SGDW', help='Using GD for update')
    parser.add_argument('--iteration', type=int, default=50, help='Maximum update iterations if not exit automatically')
    parser.add_argument('--gamma', type=float, default=0.1, help='Penalty term for logistic regression')

    args = parser.parse_args()
    return args