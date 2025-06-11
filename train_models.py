import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# Simulated data for initial training
def generate_simulated_data():
    # [rep_speed, angle_deviation, tremor, count] -> fatigue (0-1)
    X_fatigue = np.random.rand(100, 4) * [3, 30, 0.1, 20]
    y_fatigue = (X_fatigue[:, 0] + X_fatigue[:, 1] / 30 + X_fatigue[:, 2] * 10) / 4
    # [angle, speed_consistency, range_of_motion] -> quality (0-1)
    X_quality = np.random.rand(100, 3) * [180, 2, 50]
    y_quality = (X_quality[:, 0] / 180 + X_quality[:, 1] / 2 + X_quality[:, 2] / 50) / 3
    return X_fatigue, y_fatigue, X_quality, y_quality

# Train LSTM + Random Forest fatigue model
def train_fatigue_model(X, y, version):
    lstm = Sequential([
        LSTM(50, input_shape=(X.shape[1], 1), return_sequences=False),
        Dense(1, activation='sigmoid')
    ])
    lstm.compile(optimizer='adam', loss='mse')
    X_lstm = X.reshape(X.shape[0], X.shape[1], 1)
    lstm.fit(X_lstm, y, epochs=10, verbose=0)
    rf = RandomForestClassifier()
    rf.fit(X, y > 0.5)
    model = {'lstm': lstm, 'rf': rf}
    with open(f'models/fatigue_model_v{version}.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

# Train Random Forest quality model
def train_quality_model(X, y, version):
    rf = RandomForestClassifier()
    rf.fit(X, y > 0.5)
    with open(f'models/quality_model_v{version}.pkl', 'wb') as f:
        pickle.dump(rf, f)
    return rf

# Update models with new data
def update_models(new_data_fatigue, new_data_quality):
    version = len([f for f in os.listdir('models') if 'fatigue_model' in f]) + 1
    X_fatigue, y_fatigue, X_quality, y_quality = generate_simulated_data()
    # Append new data
    X_fatigue = np.vstack([X_fatigue, new_data_fatigue[:, :-1]])
    y_fatigue = np.append(y_fatigue, new_data_fatigue[:, -1])
    X_quality = np.vstack([X_quality, new_data_quality[:, :-1]])
    y_quality = np.append(y_quality, new_data_quality[:, -1])
    train_fatigue_model(X_fatigue, y_fatigue, version)
    train_quality_model(X_quality, y_quality, version)

if __name__ == "__main__":
    X_fatigue, y_fatigue, X_quality, y_quality = generate_simulated_data()
    train_fatigue_model(X_fatigue, y_fatigue, 1)
    train_quality_model(X_quality, y_quality, 1)
