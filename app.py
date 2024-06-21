import os
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import requests
from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

app = Flask(__name__)

# Define paths for model and scaler
model_path = 'saved_models/crypto_price_prediction_model.keras'
scaler_path = 'saved_models/crypto_price_prediction_scaler.pkl'


# Define SQLite database
engine = create_engine('sqlite:///crypto_predictions.db', echo=True)
Base = declarative_base()

# Define ORM model for predictions
class Prediction(Base):
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    prediction_value = Column(Float)

Base.metadata.create_all(engine)

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

# Function to make predictions
def make_predictions(model, data, scaler, window_size, num_predictions):
    predictions = []

    prices = [entry['close'] for entry in data[-window_size:]]
    prices_normalized = scaler.transform(np.array(prices).reshape(-1, 1))
    last_sequence = np.reshape(prices_normalized, (1, window_size, 1))

    for _ in range(num_predictions):
        prediction = model.predict(last_sequence)[0, 0]
        predictions.append(prediction)
        # Reshape last_sequence correctly after appending prediction
        last_sequence = np.append(last_sequence[:, 1:, :], np.array([[prediction]]).reshape(1, 1, 1), axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

@app.route('/predict', methods=['GET'])
def predict():
    data = fetch_crypto_data()

    if data:
        window_size = 10  # Adjust according to your model's window size
        num_predictions = 10  # Number of predictions you want to return

        # Preprocess data
        X, y, scaler = preprocess_data(data, window_size)

        # Load model and scaler for prediction
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        # Make predictions
        predictions = make_predictions(model, data, scaler, window_size, num_predictions)

        # Store predictions in SQLite database
        store_predictions(predictions)

        # Format predictions
        predictions_list = predictions.tolist()
        prediction_dict = {f"Prediction {i+1}": predictions_list[i] for i in range(len(predictions_list))}

        return jsonify(prediction_dict)
    else:
        return jsonify({"error": "Failed to fetch data. Check your internet connection or API availability."})

def store_predictions(predictions):
    # Create session to interact with database
    Session = sessionmaker(bind=engine)
    session = Session()

    # Store each prediction in the database
    for prediction_value in predictions:
        prediction = Prediction(prediction_value=prediction_value)
        session.add(prediction)

    # Commit changes to the database
    session.commit()
    session.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
