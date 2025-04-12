import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_data(file_path):
    """
    Load and preprocess the stock data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file containing stock data.

    Returns:
    tuple: Preprocessed data (scaled prices, scaler object).
    """
    # Load data
    data = pd.read_csv(file_path)
    
    # Ensure the 'Date' column is parsed as a datetime object
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    
    # Select the 'Close' prices for prediction
    prices = data['Close'].values.reshape(-1, 1)

    # Scale prices to the range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    return scaled_prices, scaler

def create_time_series(data, time_steps):
    """
    Create time-series data for training/testing.

    Parameters:
    data (array-like): Scaled data.
    time_steps (int): Number of previous time steps to use as input.

    Returns:
    tuple: Features (X) and targets (y).
    """
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)
