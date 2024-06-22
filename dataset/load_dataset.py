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

class InstallData:
    """
    - Keys:
        - url: str | URL of the dataset
        - target_folder: str | Target folder to download the dataset
    """
    def __init__(self, url: str, target_folder: str):
        self.url = url
        self.zip = None
        self.target_folder = target_folder
        # self.data = pd.read_csv("/home/zephyrus/WSL-Projects/spotify-problem/dataset/spotify-2023.csv", encoding='utf-8', encoding_errors='ignore')
        
    def downloadZipfile(self):
        try:
            if "predicting-hiring-decisions-in-recruitment-data.zip" not in os.listdir(self.target_folder):
                os.makedirs(self.target_folder, exist_ok=True)
                os.chdir(self.target_folder)
                self.zip = subprocess.run(self.url, shell=True, check=True)
                return self.zip
            else:
                print("[INFO] Zip file already exists in the directory")
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
                print("[INFO] Zip file removed after extracting successfully")
            else:
                print(f"[INFO] File already exists in the directory")
        else:
            print("[ERROR] Zip file not found in the directory")


class DataCleaning:
    def __init__(self):
        self.data = pd.read_csv("/home/zephyrus/WSL-Projects/spotify-problem/dataset/recruitment_data.csv", encoding='utf-8', encoding_errors='ignore')

    def checkData(self):
        null_data = self.data.isna().sum()
        null_percentage = self.data.isna().sum() / len(self.data)
        null_percentage = null_percentage.apply(lambda x: f"{x:.1%}")
        total_duplicated = self.data.duplicated().sum()
        print(f"\n============ Data info ============\n{null_data}\n--------------------------------\nNull percentage:\n{null_percentage}")
        print("---" * 10)
        print(f"Total missing values:  {sum(null_data)}\nTotal duplicated data: {total_duplicated}")
        print("---" * 10)

    def clearData(self):
        if self.data.isna().sum().any():
            self.data.dropna(axis=1, inplace=True)
            print("[INFO] Dropped NaN values successfully")
        elif self.data.duplicated().sum() > 0:
            self.data.drop_duplicates(inplace=True)
            print("[INFO] Dropped duplicates values successfully")
        else:
            print("[INFO] No missing values found in the dataset")
            return self.data
        # Write the cleaned DataFrame back to the CSV file
        self.data.to_csv("/home/zephyrus/WSL-Projects/spotify-problem/dataset/recruitment_data.csv", index=False, encoding='utf-8')
        print("[INFO] Cleaned data has been written back to the CSV file.")

        return self.data

if __name__ == "__main__":
    # Downloading dataset and unzipping
    installer = InstallData("kaggle datasets download -d rabieelkharoua/predicting-hiring-decisions-in-recruitment-data",
                             "/home/zephyrus/WSL-Projects/spotify-problem/dataset/")
    installer.downloadZipfile()
    installer.unzipFile()
    # import sys; sys.exit(0)
    
    # Cleaning data
    cleaner = DataCleaning()
    cleaner.checkData()
    cleaner.clearData()