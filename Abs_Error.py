import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt

# Loading dataset
dataset = pd.read_csv("Merged_IPC_6.csv")

# Splitting dataset into features (X) and target variable (y)
X = dataset[['Cumulative Instructions', 'IPC_19']]
y = dataset['IPC_03']

# Defining test size
test_size = 0.5

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape input data for time series prediction
X_train_ts = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_ts = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Initializing neural network model
model = Sequential([
    LSTM(128, activation='relu', input_shape=(X_train_ts.shape[1], X_train_ts.shape[2])),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

# Train the model
history = model.fit(X_train_ts, y_train, epochs=100, batch_size=64, validation_split=0.1, verbose=0)

# Evaluate the model
mse = model.evaluate(X_test_ts, y_test)
print(f"Mean Squared Error for test size {test_size}: {mse}")

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title(f'Training and Validation Loss (Test Size: {test_size})')
plt.show()

# Plot predicted vs actual values
predicted_IPC_03 = model.predict(X_test_ts)

# Calculate predicted Speed_up
predicted_Speed_up = predicted_IPC_03 / X_test['IPC_19'].values[:, np.newaxis]

# Sort the data by Cumulative Instructions for better visualization
sorted_indices = X_test['Cumulative Instructions'].argsort()
X_test_sorted = X_test.iloc[sorted_indices]
predicted_Speed_up_sorted = predicted_Speed_up[sorted_indices]
actual_Speed_up_sorted = dataset['Speed_up'].iloc[X_test_sorted.index]

# Calculate absolute error between predicted and actual speedup
absolute_errors = np.abs(predicted_Speed_up_sorted.flatten() - actual_Speed_up_sorted.values)

# Calculate mean and standard deviation of absolute errors
mean_error = np.mean(absolute_errors)
std_error = np.std(absolute_errors)

print(f"Mean Absolute Error: {mean_error}")
print(f"Standard Deviation of Absolute Error: {std_error}")

# Plot absolute error vs frequency
plt.figure(figsize=(10, 6))
plt.hist(absolute_errors, bins=30, edgecolor='black')
plt.xlabel('Absolute Error', fontsize=14, labelpad=20)
plt.ylabel('Frequency', fontsize=14, labelpad=20)
plt.title('Absolute Error vs Frequency for Speedup Prediction', fontsize=16)
plt.grid(True)
# Set font size for ticks
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
# Set the x-axis limits to start from 0
plt.xlim(left=0)
plt.show()
