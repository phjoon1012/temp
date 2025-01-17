# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Load your dataset
file_path = "model_data.csv"  # Replace with your file name
btc_data = pd.read_csv(file_path)

# Rename columns for clarity
btc_data.columns = [
    "Open", "High", "Low", "Close", "Volume"

]


# Convert timestamps to datetime format
btc_data["Open"] = pd.to_datetime(btc_data["Open"], unit='ms')
btc_data["Close"] = pd.to_datetime(btc_data["Close"], unit='ms')

# Drop unnecessary columns

#
# Normalize data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(btc_data)

# Prepare sequences for time-series forecasting
time_steps = 60  # Use the last 60 time steps to predict the next one
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps, 3])  # Predict the "Close" value
    return np.array(X), np.array(y)

X, y = create_sequences(normalized_data, time_steps)

# Split the data into training and testing sets
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss}")

# Make predictions
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(
    np.hstack((np.zeros((len(predicted_prices), 4)), predicted_prices))
)[:, -1]  # Reverse scaling only for the Close price

# Plot predictions vs actual
actual_prices = scaler.inverse_transform(
    np.hstack((np.zeros((len(y_test), 4)), y_test.reshape(-1, 1)))
)[:, -1]


def predict_future(model, last_sequence, num_predictions, scaler):
    """
    Predict future prices using the trained model.
    - model: Trained LSTM model.
    - last_sequence: The last sequence of data from the training/testing set.
    - num_predictions: Number of future predictions to make.
    - scaler: MinMaxScaler used for scaling data.
    """
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(num_predictions):
        # Predict the next value
        next_pred = model.predict(current_sequence[np.newaxis, :, :], verbose=0)[0][0]
        future_predictions.append(next_pred)

        # Update the sequence with the predicted value
        next_sequence = np.append(current_sequence[1:], [[0, 0, 0, next_pred, 0]], axis=0)
        current_sequence = next_sequence

    # Reverse scale the predictions
    future_predictions_scaled = scaler.inverse_transform(
        np.hstack((np.zeros((len(future_predictions), 4)), np.array(future_predictions).reshape(-1, 1)))
    )[:, -1]
    return future_predictions_scaled

# Use the last sequence of the test set to start the predictions
last_sequence = X_test[-1]
num_future_predictions = 24  # Predict for the next 24 hours

future_prices = predict_future(model, last_sequence, num_future_predictions, scaler)

def create_lag_features(data, lags=60):
    features = {}
    for lag in range(1, lags + 1):
        features[f"lag_{lag}"] = data.shift(lag)
    return pd.DataFrame(features)

lags = 60
lagged_data = create_lag_features(pd.Series(normalized_data[:, 3]), lags)  # Use the normalized "Close" column
lagged_data["target"] = normalized_data[:, 3]  # Add target as the "Close" value
lagged_data.dropna(inplace=True)

# Split Gradient Boosting data into features and target
X_gb = lagged_data.drop(columns=["target"])
y_gb = lagged_data["target"]

# Split into training and testing sets
X_gb_train, X_gb_test, y_gb_train, y_gb_test = train_test_split(X_gb, y_gb, test_size=0.2, shuffle=False)

# 2. Train the Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_gb_train, y_gb_train)

# 3. Make predictions using Gradient Boosting
gb_pred = gb_model.predict(X_gb_test)

# Inverse transform Gradient Boosting predictions to match actual prices
gb_pred_inverse = scaler.inverse_transform(
    np.hstack((np.zeros((len(gb_pred), 4)), gb_pred.reshape(-1, 1)))
)[:, -1]

def predict_future_gb(model, last_sequence, num_predictions):
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(num_predictions):
        # Predict the next value
        next_pred = model.predict(current_sequence.reshape(1, -1))[0]
        future_predictions.append(next_pred)
        
        # Update the sequence by appending the predicted value and removing the oldest value
        current_sequence = np.append(current_sequence[1:], next_pred)  # Keep 60 features
    
    return future_predictions

# 4. Align LSTM predictions with Gradient Boosting predictions
lstm_pred_inverse = predicted_prices[-len(gb_pred):]  # Ensure the same length as gb_pred

# 5. Combine LSTM and Gradient Boosting predictions
combined_pred = (lstm_pred_inverse + gb_pred_inverse) / 2  # Simple averaging

# 6. Plot Actual vs Predicted Prices (Combined)
plt.figure(figsize=(12, 6))

# Plot actual prices
plt.plot(actual_prices[-len(gb_pred):], label="Actual Prices", linestyle='-', color='blue')

# Plot LSTM predictions
plt.plot(lstm_pred_inverse, label="LSTM Predictions", linestyle='--', color='green')

# Plot Gradient Boosting predictions
plt.plot(gb_pred_inverse, label="Gradient Boosting Predictions", linestyle='--', color='orange')

# Plot combined predictions
plt.plot(combined_pred, label="Combined Predictions", linestyle='-', color='red')

# Add titles and labels
plt.title("BTC Price Prediction (Combined LSTM and Gradient Boosting)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

lstm_future_pred = predict_future(model, last_sequence, num_future_predictions, scaler)

# Predict future prices with Gradient Boosting
last_sequence_gb = X_gb_test.iloc[-1].values  # Last sequence for Gradient Boosting
gb_future_pred = predict_future_gb(gb_model, last_sequence_gb, num_future_predictions)

# Combine future predictions
lstm_future_pred = scaler.inverse_transform(
    np.hstack((np.zeros((len(lstm_future_pred), 4)), np.array(lstm_future_pred).reshape(-1, 1)))
)[:, -1]

gb_future_pred = scaler.inverse_transform(
    np.hstack((np.zeros((len(gb_future_pred), 4)), np.array(gb_future_pred).reshape(-1, 1)))
)[:, -1]


# Combine future predictions
combined_future_pred = (lstm_future_pred + gb_future_pred) / 2

combined_future_pred = (lstm_future_pred + gb_future_pred) / 2  # Simple averaging

plt.figure(figsize=(12, 6))

# Plot LSTM future predictions
plt.plot(
    range(len(actual_prices), len(actual_prices) + len(lstm_future_pred)),
    lstm_future_pred,
    label="LSTM Future Predictions",
    linestyle='--',
    color='green'
)

# Plot Gradient Boosting future predictions
plt.plot(
    range(len(actual_prices), len(actual_prices) + len(gb_future_pred)),
    gb_future_pred,
    label="Gradient Boosting Future Predictions",
    linestyle='--',
    color='orange'
)

# Plot combined future predictions
plt.plot(
    range(len(actual_prices), len(actual_prices) + len(combined_future_pred)),
    combined_future_pred,
    label="Combined Future Predictions",
    linestyle='-',
    color='red'
)

# Add titles and labels
plt.title("BTC Future Price Prediction (Combined LSTM and Gradient Boosting)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
