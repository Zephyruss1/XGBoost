"""
---------------------------------
Author: Zephyrus                |
Date: 06.22.2024                |
Purpose: Run the project        |
---------------------------------
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import time 
from dataset.load_dataset import LoadDataset
from xgboost import XGBoost
from options import get_options
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

args = get_options()

def main_run():
    if __name__ == '__main__':
        dataset_loader = LoadDataset()  # Step 1: Instantiate the class
        (X_train, X_test), (y_train, y_test) = dataset_loader.splitData()  # Step 2: Call the method on the instance

        model = XGBoost(args=args, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        print(f"[ARGS INFO] Learning Rate: {args.lr} | Max Depth: {args.max_depth} | N-Estimators: {args.n_estimators}")
        print("---" * 10)

        for i in range(args.n_estimators):
            t0 = time.time()
            model.fit_single_iteration()
            t1 = time.time()
            total_ms  = (t1 - t0) * 1000
            print(f"[RUN INFO] Iteration: {i+1} | {total_ms:.2f} ms | MSE: {model.mse():.2f} | R2 Score: {model.r2_score():.2f}")

main_run()
