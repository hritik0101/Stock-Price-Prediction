import numpy as np
from sklearn.model_selection import train_test_split
from model import build_fnn, build_rnn_lstm
from data_loader import create_time_series
import matplotlib.pyplot as plt

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train the given model.

    Parameters:
    model: Compiled model to train.
    X_train, y_train: Training data and labels.
    X_val, y_val: Validation data and labels.
    epochs (int): Number of training epochs.
    batch_size (int): Batch size.

    Returns:
    history: Training history.
    """
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history

def plot_training_history(history):
    """
    Plot training and validation loss.

    Parameters:
    history: Training history returned by model.fit.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Example usage (assuming data is already preprocessed):
    data, _ = load_data("path_to_data.csv")  # Replace with actual data file
    time_steps = 60
    X, y = create_time_series(data, time_steps)

    # Reshape for RNN (samples, time_steps, features)
    X_rnn = X.reshape((X.shape[0], X.shape[1], 1))

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X_rnn, y, test_size=0.2, random_state=42)

    # Build and train model
    rnn_model = build_rnn_lstm(input_shape=(time_steps, 1))
    history = train_model(rnn_model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

    # Plot loss
    plot_training_history(history)

    # Save the trained model
    rnn_model.save("rnn_model.h5")
