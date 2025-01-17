import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import matplotlib.pyplot as plt

# Load and preprocess data
def load_and_preprocess_data(file_path):
    btc_data = pd.read_csv(file_path)
    btc_data.columns = [
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Base Asset Volume", "Number of Trades",
        "Taker Buy Volume", "Taker Buy Base Asset Volume", "Ignore"
    ]
    btc_data["Open Time"] = pd.to_datetime(btc_data["Open Time"], unit='ms')
    btc_data["Close Time"] = pd.to_datetime(btc_data["Close Time"], unit='ms')
    btc_data = btc_data.drop(columns=["Ignore"])

    return btc_data[["Open", "High", "Low", "Close", "Volume"]]


# Create sequences for LSTM
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps, 3])  # Predict "Close" value
    return np.array(X), np.array(y)

# Predict future prices
def predict_future(model, last_sequence, num_predictions, scaler):
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(num_predictions):
        next_pred = model.predict(current_sequence[np.newaxis, :, :], verbose=0)[0][0]
        future_predictions.append(next_pred)
        
        # Use the predicted "Close" value and replicate the rest of the sequence
        next_sequence = np.roll(current_sequence, -1, axis=0)
        next_sequence[-1, :] = [next_pred, current_sequence[-1, 1], current_sequence[-1, 2], 
                                current_sequence[-1, 3], current_sequence[-1, 4]]
        current_sequence = next_sequence

    # Scale only the "Close" column for inverse transformation
    future_predictions_scaled = scaler.inverse_transform(
        np.hstack((np.zeros((len(future_predictions), 4)), np.array(future_predictions).reshape(-1, 1)))
    )[:, -1]

    return future_predictions_scaled

# Main pipeline


# file_path = "model_data.csv"
file_path = "cleaned_adjusted_data.csv"
data = load_and_preprocess_data(file_path)

# Normalize data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

# Prepare data for LSTM
time_steps = 60
X, y = create_sequences(normalized_data, time_steps)
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build the LSTM model
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Make predictions
y_pred = model.predict(X_test, verbose=0)
predicted_prices = scaler.inverse_transform(
    np.hstack((np.zeros((len(y_pred), 4)), y_pred))
)[:, -1]

# Actual prices for comparison
actual_prices = scaler.inverse_transform(
    np.hstack((np.zeros((len(y_test), 4)), y_test.reshape(-1, 1)))
)[:, -1]

# Predict future prices
num_future_predictions = 30
future_predictions_scaled = predict_future(model, X_test[-1], num_future_predictions, scaler)

# Generate time index for future predictions
future_index = pd.date_range(start=data.index[-1], periods=num_future_predictions + 1, freq="1H")[1:]

bias = np.mean(actual_prices - predicted_prices)
corrected_predictions = predicted_prices + bias
corrected_future_predictions = future_predictions_scaled + bias

# Plot results
plt.figure(figsize=(14, 7))

# Plot actual prices
plt.plot(actual_prices, label="Actual Prices", color='blue')

# Plot corrected predictions
plt.plot(corrected_predictions, label="Corrected Predictions", color='green', linestyle='--')

# Plot future predictions
plt.plot(
    range(len(actual_prices), len(actual_prices) + len(corrected_future_predictions)),
    corrected_future_predictions,
    label="Corrected Future Predictions",
    linestyle="--",
    color="orange"
)

# Customize plot
plt.title("BTC Price Prediction with Bias Correction", fontsize=16)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Price", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

results_df = pd.DataFrame({
    "Actual Prices": actual_prices,
    "Predicted Prices": predicted_prices,
    "Future Predictions": list(future_predictions_scaled) + [None] * (len(actual_prices) - len(future_predictions_scaled))
})

