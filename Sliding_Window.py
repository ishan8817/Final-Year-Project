import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Loading dataset
dataset = pd.read_csv("Merged_IPC_6.csv")

# Extracting features and target variable
X = dataset[['Cumulative Instructions', 'IPC_19']].values
y = dataset['IPC_03'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define window size
window_size = 100  # You can adjust this window size as needed

# Create sequences of input-output pairs using sliding window approach
X_seq, y_seq = [], []
for i in range(len(X_scaled) - window_size):
    X_seq.append(X_scaled[i:i + window_size])
    y_seq.append(y[i + window_size])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# Initializing neural network model
model = Sequential([
    LSTM(128, activation='relu', input_shape=(X_seq.shape[1], X_seq.shape[2])),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

# Train the model
history = model.fit(X_seq, y_seq, epochs=100, batch_size=64, validation_split=0.1, verbose=1)

# Plot training and validation loss
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch', fontsize=14, labelpad=20)
plt.ylabel('Loss', fontsize=14, labelpad=20)
plt.legend()
plt.title(f'Training and Validation Loss', fontsize=14)
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.grid(True)
plt.show()

# Predict IPC_03 values using the trained model
predicted_IPC_03 = model.predict(X_seq)

# Convert to numpy array
predicted_IPC_03 = np.array(predicted_IPC_03)


# Plot actual vs predicted IPC_03 values
plt.figure(figsize=(10, 6))
plt.plot(X[window_size:, 0], y_seq, label='Actual IPC P(big) Core', color='purple')
plt.plot(X[window_size:, 0], predicted_IPC_03, label='Predicted IPC P(big) Core', color='red')
plt.xlabel('Cumulative Instructions ($\\times10^{11}$)', fontsize=14, labelpad=20)
plt.ylabel('Instructions per Cycle (IPC)', fontsize=14, labelpad=20)
plt.title(f'IPC Predictions vs Cumulative Instructions', fontsize=16)
plt.legend()
# Set the x-axis & y-axis limits to start from 0
plt.xlim(left=X[window_size:, 0].min())
plt.ylim(bottom=0)
plt.grid(True)
# Set font size for ticks
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
# Apply scientific notation formatting to x-axis ticks
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1e11:.0f}"))
plt.show()

# Calculate predicted Speed_up
#predicted_Speed_up = predicted_IPC_03.squeeze() / X[window_size:, 1]

# Calculate predicted Speed_up
predicted_Speed_up = []
for i in range(len(X) - window_size):
    predicted_IPC_window = predicted_IPC_03[i:i+window_size].mean()
    predicted_Speed_up.append(predicted_IPC_window / X[i + window_size, 1])

# Plot actual and predicted Speed_up
actual_Speed_up = dataset['Speed_up'].values[window_size:]

# Plot actual and predicted Speed_up
plt.figure(figsize=(12, 8))
plt.plot(X[window_size:, 0], actual_Speed_up, label='Actual Speed_up', color='skyblue')
plt.plot(X[window_size:, 0], predicted_Speed_up, label='Predicted Speed_up', color='red')
plt.xlabel('Cumulative Instructions ($\\times10^{11}$)', fontsize=14, labelpad=20)
plt.ylabel('Speedup', fontsize=14, labelpad=20)
plt.title(f'Actual vs Predicted Speedup', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
# Set the x-axis limits to start from 0
plt.xlim(left=X[window_size:, 0].min())
# Set font size for ticks
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
# Apply scientific notation formatting to x-axis ticks
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1e11:.0f}"))
plt.show()

