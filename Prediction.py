import numpy as np
from tensorflow.keras.models import load_model
from data_loader import create_time_series
import matplotlib.pyplot as plt

def make_prediction(model, data, time_steps, scaler):
    """
    Use the model to make predictions on the given data.

    Parameters:
    model: Trained model.
    data (array-like): Scaled input data.
    time_steps (int): Number of time steps used for prediction.
    scaler: Scaler used for normalizing the data.

    Returns:
    array-like: Predicted values (unscaled).
    """
    X, _ = create_time_series(data, time_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for RNN input

    predictions = model.predict(X)
    return scaler.inverse_transform(predictions)  # Rescale predictions

def plot_predictions(actual, predicted):
    """
    Plot actual vs. predicted stock prices.

    Parameters:
    actual (array-like): Actual stock prices.
    predicted (array-like): Predicted stock prices.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual Prices', color='blue')
    plt.plot(predicted, label='Predicted Prices', color='red')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Example usage:
    from data_loader import load_data

    # Load data
    data, scaler = load_data("path_to_data.csv")  # Replace with actual data file
    time_steps = 60

    # Load trained model
    model = load_model("rnn_model.h5")

    # Make predictions
    predictions = make_prediction(model, data, time_steps, scaler)

    # Plot predictions
    actual_prices = scaler.inverse_transform(data[time_steps:])  # Unscaled actual prices
    plot_predictions(actual_prices, predictions)
