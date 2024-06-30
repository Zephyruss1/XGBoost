"""
---------------------------------
Author: Zephyrus                |
Date: 06.22.2024                |
Purpose: Load Dataset           |
---------------------------------
"""

import subprocess
import os
import pandas as pd
import numpy as np


class InstallData:
    """
    - Keys:
            1) url: str | URL of the dataset
            2) target_folder: str | Target folder to download the dataset
            3) zip: file | Initialize zip file
    """

    def __init__(self, url: str, target_folder: str):
        self.url = url
        self.zip = None
        self.target_folder = target_folder

    def downloadZipfile(self):
        try:
            if "predicting-hiring-decisions-in-recruitment-data.zip" not in os.listdir(self.target_folder):
                os.makedirs(self.target_folder, exist_ok=True)
                os.chdir(self.target_folder)
                self.zip = subprocess.run(self.url, shell=True, check=True)
                return self.zip
            else:
                print("[DATA INFO] Zip file already exists in the directory")
        except Exception as e:
            print(f"[ERROR] A error found. {e}")

    def unzipFile(self):
        try:
            import zipfile
        except ImportError:
            raise ImportError("zipfile module is not installed. Please install it using 'pip install zipfile'")

        if "predicting-hiring-decisions-in-recruitment-data.zip" in os.listdir(self.target_folder):
            if "recruitment_data.csv" not in os.listdir(self.target_folder):
                zip_folder = os.path.join(self.target_folder, "predicting-hiring-decisions-in-recruitment-data.zip")
                with zipfile.ZipFile(zip_folder) as zipfile:
                    zipfile.extractall(self.target_folder)
                os.remove(zip_folder)
                print("[DATA INFO] Zip file removed after extracting successfully")
            else:
                print(f"[DATA INFO] File already exists in the directory")
        else:
            print("[ERROR] Zip file not found in the directory")


class DataCleaning:
    """
    - Keys:
            1) data: csv | The dataset
    """

    def __init__(self):
        self.data = pd.read_csv("/home/zephyrus/WSL-Projects/spotify-problem/dataset/recruitment_data.csv",
                                encoding='utf-8')

    def checkData(self):
        null_data = self.data.isna().sum()
        null_percentage = self.data.isna().sum() / len(self.data)
        null_percentage = null_percentage.apply(lambda x: f"{x:.1%}")
        total_duplicated = self.data.duplicated().sum()
        print(
            f"\n============ Data info ============\n{null_data}\n--------------------------------\nNull percentage:\n{null_percentage}")
        print("---" * 10)
        print(f"Total missing values:  {sum(null_data)}\nTotal duplicated data: {total_duplicated}")
        print("---" * 10)

    def clearData(self):
        if self.data.isna().sum().any():
            self.data.dropna(axis=1, inplace=True)
            print("[Downloader Info] Dropped NaN values successfully")
        elif self.data.duplicated().sum() > 0:
            self.data.drop_duplicates(inplace=True)
            print("[Downloader Info] Dropped duplicates values successfully")
        else:
            print("[Downloader Info] No missing values found in the dataset")
            return self.data
        # Write the cleaned DataFrame back to the CSV file
        self.data.to_csv("/home/zephyrus/WSL-Projects/spotify-problem/dataset/recruitment_data.csv", index=False,
                         encoding='utf-8')
        print("[Downloader Info] Cleaned data has been written back to the CSV file.")

        return self.data


class LoadDataset:
    """
    - Keys:
            1) data: csv | The dataset
    """

    def __init__(self):
        self.data = DataCleaning().clearData()

    def splitData(self, test_size=0.2, random_state=42):
        # Ensure reproducibility
        np.random.seed(random_state)

        # Generate shuffled indices
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)

        # Calculate split index
        split_idx = int(len(self.data) * (1 - test_size))

        # Split indices
        train_indices, test_indices = indices[:split_idx], indices[split_idx:]

        # Splitting features
        X = self.data.drop('HiringDecision', axis=1)
        y = self.data['HiringDecision']

        # Use indices to create train/test splits
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        print("---" * 10)
        print(f"X_train: {X_train.shape}, X_test: {X_test.shape}\ny_train: {y_train.shape}, y_test: {y_test.shape}")
        print("---" * 10)
        return (X_train, X_test), (y_train, y_test)

    def standardScaler(self):
        # Using Standard Scaler formula
        (X_train, X_test), (y_train, y_test) = self.splitData()
        X_train_scaled = (X_train - X_train.mean()) / X_train.std()
        X_test_scaled = (X_test - X_test.mean()) / X_test.std()
        print("[DATA INFO] Data has been scaled successfully")
        print("---" * 10)
        return (X_train_scaled, X_test_scaled), (y_train, y_test)


if __name__ == "__main__":
    # Downloading dataset and unzipping
    installer = InstallData(
        "kaggle datasets download -d rabieelkharoua/predicting-hiring-decisions-in-recruitment-data",
        "/home/zephyrus/WSL-Projects/spotify-problem/dataset/")
    installer.downloadZipfile()
    installer.unzipFile()

    # Cleaning data
    cleaner = DataCleaning()
    cleaner.checkData()
    cleaner.clearData()

    # Splitting data
    load = LoadDataset()
    # Standard Scale
    load.standardScaler()
