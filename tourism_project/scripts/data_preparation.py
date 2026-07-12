import pandas as pd
import os

DATA_FILE_PATH = 'tourism_project/data/tourism.csv'

def prepare_data():
    if not os.path.exists(DATA_FILE_PATH):
        print(f"Error: Dataset file not found at {DATA_FILE_PATH}. Ensure the register-dataset job ran successfully.")
        exit(1)

    print(f"Loading data from {DATA_FILE_PATH}...")
    df = pd.read_csv(DATA_FILE_PATH)
    print("Data loaded successfully. Displaying info and head:")
    df.info()
    print(df.head())
    print("Data preparation job completed.")

if __name__ == '__main__':
    prepare_data()
