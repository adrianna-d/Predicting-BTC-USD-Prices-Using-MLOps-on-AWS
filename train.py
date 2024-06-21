import argparse
import mlflow
import mlflow.keras
import numpy as np
import requests
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Initialize MLFlow at the beginning of your script
mlflow.set_tracking_uri('http://localhost:5000')  # Set your MLFlow tracking URI
mlflow.set_experiment('Crypto_Price_Prediction_GRU')  # Set your MLFlow experiment name

# Function to fetch cryptocurrency data
def fetch_crypto_data():
    try:
        url = 'https://min-api.cryptocompare.com/data/v2/histohour'
        params = {
            'fsym': 'BTC',
            'tsym': 'USD',
            'limit': 2000,   # Number of data points
            'aggregate': 1,  # Hourly data
        }
        response = requests.get(url, params=params)
        data = response.json()['Data']['Data']  # Extracting the historical data
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    
# Function to preprocess data
def preprocess_data(data, window_size):
    if not isinstance(data, list) or not isinstance(data[0], dict):
        raise ValueError("Input data must be a list of dictionaries")

    prices = [entry['close'] for entry in data]

    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_normalized = scaler.fit_transform(np.array(prices).reshape(-1, 1))

    X, y = [], []
    for i in range(len(prices_normalized) - window_size):
        X.append(prices_normalized[i:i + window_size, 0])
        y.append(prices_normalized[i + window_size, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

# Function to build GRU model
def build_gru_model(input_shape, gru_units, dropout_rate):
    model = Sequential()
    model.add(GRU(units=gru_units, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to make predictions
def make_predictions(model, data, scaler, window_size, num_predictions):
    predictions = []
    
    # Extract the last window_size data points from the original data (not from X)
    prices = [entry['close'] for entry in data[-window_size:]]
    prices_normalized = scaler.transform(np.array(prices).reshape(-1, 1))
    last_sequence = np.reshape(prices_normalized, (1, window_size, 1))
    
    for _ in range(num_predictions):
        prediction = model.predict(last_sequence)[0, 0]
        predictions.append(prediction)
        last_sequence = np.append(last_sequence[:, 1:, :], [[prediction]], axis=1)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

# Function to evaluate model
def evaluate_model(model, X, y, scaler):
    # Predict on X
    y_pred = model.predict(X)
    
    # Inverse transform to get actual values
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_inv = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_inv, y_pred_inv)
    mae = mean_absolute_error(y_inv, y_pred_inv)
    
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Mean Absolute Error (MAE): {mae}')

    return mae  # Return MAE for logging

# Main function for training and logging with MLFlow
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add arguments for hyperparameters
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--gru_units', type=int, default=50)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)

    args, _ = parser.parse_known_args()

    # Start MLFlow run
    with mlflow.start_run(run_name='Crypto_Price_Prediction_GRU'):
        # Log parameters
        mlflow.log_param('window_size', args.window_size)
        mlflow.log_param('gru_units', args.gru_units)
        mlflow.log_param('dropout_rate', args.dropout_rate)
        mlflow.log_param('epochs', args.epochs)
        mlflow.log_param('batch_size', args.batch_size)

        # Fetch data
        crypto_data = fetch_crypto_data()

        # Preprocess data
        X, y, scaler = preprocess_data(crypto_data, args.window_size)

    
        # Build GRU model
        input_shape = (X.shape[1], 1)
        model = build_gru_model(input_shape, args.gru_units, args.dropout_rate)

        # Train model
        history = model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size, verbose=1)

        # Evaluate model
        mae=evaluate_model(model, X, y, scaler)

        # Log metrics
        mlflow.log_metric('final_mse', history.history['loss'][-1])
        mlflow.log_metric('final_mae', mae)

        # Save model artifact to MLFlow
        mlflow.keras.log_model(model, 'crypto_price_prediction_model')

        # Save scaler artifact
        joblib.dump(scaler, 'crypto_price_prediction_scaler.pkl')

        print("Model training and logging completed successfully.")
