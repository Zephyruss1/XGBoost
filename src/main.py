"""
---------------------------------
Author: Zephyrus                |
Date: 06.22.2024                |
Purpose: Run the project        |
---------------------------------
"""

import sys
"------------------------------------------------------------------"
sys.append("/home/zephyrus/WSL-Projects/spotify-problem/src")
sys.append("/home/zephyrus/WSL-Projects/spotify-problem/dataset")
"------------------------------------------------------------------"
import os
import numpy as np
import pickle as pkl
from dataset import load_dataset
from xgboost import XGBoost
from options import get_options
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    pass