import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
# Load model and scaler for prediction
model_path = 'saved_models/crypto_price_prediction_model.keras'
scaler_path = 'saved_models/crypto_price_prediction_scaler.pkl'

model = load_model(model_path)
scaler = joblib.load(scaler_path)
# Function to preprocess data
def preprocess_data(data, window_size):
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

# Function to make predictions
def make_predictions(model, data, scaler, window_size, num_predictions):
    predictions = []

    prices = [entry['close'] for entry in data[-window_size:]]
    prices_normalized = scaler.transform(np.array(prices).reshape(-1, 1))
    last_sequence = np.reshape(prices_normalized, (1, window_size, 1))

    for _ in range(num_predictions):
        prediction = model.predict(last_sequence)[0, 0]
        predictions.append(prediction)
        last_sequence = np.append(last_sequence[:, 1:, :], [[prediction]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    data = content['data']  # Assuming data is passed as JSON with key 'data'
    window_size = 10  # Adjust as per your model's window size
    num_predictions = 10  # Adjust as needed

    # Preprocess data
    X, y, scaler = preprocess_data(data, window_size)

    # Make predictions
    predictions = make_predictions(model, data, scaler, window_size, num_predictions)

    # Return predictions as JSON response
    return jsonify({'predictions': predictions.tolist()})
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
