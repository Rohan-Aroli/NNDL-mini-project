import os
import urllib.request
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv"
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "air_quality.csv")

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(DATA_FILE):
        print(f"Downloading dataset to {DATA_FILE}...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print("Download complete.")

def parse_datetime(x):
    return datetime.strptime(x, '%Y %m %d %H')

def load_and_preprocess_data():
    download_data()

    df = pd.read_csv(DATA_FILE)

    # Parse datetime
    df['datetime'] = df.apply(lambda row: parse_datetime(f"{int(row['year'])} {int(row['month'])} {int(row['day'])} {int(row['hour'])}"), axis=1)
    df.set_index('datetime', inplace=True)
    df.drop(['No', 'year', 'month', 'day', 'hour'], axis=1, inplace=True)
    df.columns = ['pm2.5', 'dewp', 'temp', 'pres', 'cbwd', 'iws', 'is', 'ir']
    df.index.name = 'date'

    # Preprocessing: forward-fill missing values
    df['pm2.5'] = df['pm2.5'].ffill()
    df['pm2.5'] = df['pm2.5'].fillna(0) # in case the first value is NA

    # One-hot encode the categorical wind direction feature
    # Using simple label encoding for simplicity, or dummy variables
    # Let's drop it to match purely numeric sequence or encode it. Wait, Multivariate LSTM usually needs numeric.
    # Let's map cbwd to numbers
    # wind_dir = {'NE': 1, 'NW': 2, 'SE': 3, 'cv': 4}
    # df['cbwd'] = df['cbwd'].map(wind_dir)
    # Actually one-hot encoding is better:
    # However we can just use LabelEncoding
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    df['cbwd'] = encoder.fit_transform(df['cbwd'])

    return df

class AirQualityDataset(Dataset):
    def __init__(self, data, lookback=24, horizon=1):
        self.data = data
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.lookback - self.horizon + 1

    def __getitem__(self, idx):
        # We need X as sequence of shape (lookback, features)
        # We need y as the target (pm2.5 which is at index 0 typically) for (lookback + horizon - 1)
        # Actually it says forecast horizon = 1 hour, so predict next hour's PM2.5
        X = self.data[idx : idx + self.lookback]
        y = self.data[idx + self.lookback : idx + self.lookback + self.horizon, 0] # assume pm2.5 is at index 0

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def get_dataloaders(batch_size=64, lookback=24, horizon=1):
    df = load_and_preprocess_data()

    # Train/Val/Test split: 70% / 15% / 15% (chronological, no shuffle)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    # Split data before scaling to avoid data leakage
    train_raw = df.values[:train_end]
    val_raw = df.values[train_end:val_end]
    test_raw = df.values[val_end:]

    # Normalize features with MinMaxScaler fit ONLY on training data
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_raw)
    val_data = scaler.transform(val_raw)
    test_data = scaler.transform(test_raw)

    train_dataset = AirQualityDataset(train_data, lookback, horizon)
    val_dataset = AirQualityDataset(val_data, lookback, horizon)
    test_dataset = AirQualityDataset(test_data, lookback, horizon)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler
