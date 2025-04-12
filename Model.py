import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def build_fnn(input_shape):
    """
    Build a Feed Forward Neural Network (FNN).

    Parameters:
    input_shape (tuple): Shape of the input data (features,).

    Returns:
    model: Compiled FNN model.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(1)  # Output layer
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_rnn_lstm(input_shape):
    """
    Build a Recurrent Neural Network with LSTM.

    Parameters:
    input_shape (tuple): Shape of the input data (time_steps, features).

    Returns:
    model: Compiled RNN with LSTM model.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)  # Output layer
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
