from data_loader import load_data, create_time_series
from model import build_rnn_lstm
from train import train_model, plot_training_history
from predict import make_prediction, plot_predictions
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import os

def main():
    # Load and preprocess data
    file_path = "path_to_data.csv"  # Replace with actual file path
    scaled_data, scaler = load_data(file_path)

    # Define parameters
    time_steps = 60

    # Create time-series data
    X, y = create_time_series(scaled_data, time_steps)
    X_rnn = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for RNN

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_rnn, y, test_size=0.2, random_state=42)

    # Check if model already exists
    model_path = "rnn_model.h5"
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = load_model(model_path)
    else:
        print("Training a new model...")
        model = build_rnn_lstm(input_shape=(time_steps, 1))
        history = train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
        plot_training_history(history)
        model.save(model_path)

    # Make predictions
    predictions = make_prediction(model, scaled_data, time_steps, scaler)
    actual_prices = scaler.inverse_transform(scaled_data[time_steps:])  # Unscaled actual prices

    # Plot predictions vs actual prices
    plot_predictions(actual_prices, predictions)

if __name__ == "__main__":
    main()
