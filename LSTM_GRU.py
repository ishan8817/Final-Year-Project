import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Loading dataset
dataset = pd.read_csv("Merged_IPC_6.csv")

# Splitting dataset into features (X) and target variable (y)
X = dataset[['Cumulative Instructions', 'IPC_19']]
y = dataset['IPC_03']

# Defining the test set size
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

# Initialize lists to store results
models = {'LSTM': {'model': Sequential([
                        LSTM(128, activation='relu', input_shape=(X_train_ts.shape[1], X_train_ts.shape[2])),
                        Dense(64, activation='relu'),
                        Dense(32, activation='relu'),
                        Dense(1)
                    ])},
          'GRU': {'model': Sequential([
                        GRU(128, activation='relu', input_shape=(X_train_ts.shape[1], X_train_ts.shape[2])),
                        Dense(64, activation='relu'),
                        Dense(32, activation='relu'),
                        Dense(1)
                    ])}
         }

# Compile and train models
for model_name, model_data in models.items():
    model = model_data['model']
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    history = model.fit(X_train_ts, y_train, epochs=100, batch_size=64, validation_split=0.1, verbose=0)
    model_data['history'] = history

# Evaluate models
for model_name, model_data in models.items():
    model = model_data['model']
    X_test_ts = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
    mse = model.evaluate(X_test_ts, y_test)
    print(f"Mean Squared Error for {model_name} model: {mse}")
    model_data['mse'] = mse

# Plot IPC predictions vs Cumulative Instructions for both models
plt.figure(figsize=(10, 6))
for model_name, model_data in models.items():
    model = model_data['model']
    predicted_IPC_03 = model.predict(X_test_ts)

    # Sort the test data by Cumulative Instructions
    X_test_sorted = X_test.sort_values(by='Cumulative Instructions')

    # Sort the predicted IPC_03 data by Cumulative Instructions
    predicted_IPC_03_sorted = predicted_IPC_03[np.argsort(X_test['Cumulative Instructions'])]

    # Plot IPC predictions vs Cumulative Instructions
    plt.plot(X_test_sorted['Cumulative Instructions'], predicted_IPC_03_sorted, label=f'Predicted IPC P(big) Core ({model_name})')
plt.plot(X_test_sorted['Cumulative Instructions'], y_test.values[np.argsort(X_test['Cumulative Instructions'])], label='Actual IPC P(big) Core', color='purple', zorder=1)
plt.xlabel('Cumulative Instructions ($\\times10^{11}$)', fontsize=14, labelpad=20)
plt.ylabel('IPC (Instructions per Cycle)', fontsize=14, labelpad=20)
plt.title('Predicted vs Actual IPC (LSTM vs GRU)', fontsize=16)
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1e11:.0f}"))
# Set font size for ticks
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.xlim(left=0)
plt.show()

# Calculate predicted Speed_up for both models
for model_name, model_data in models.items():
    model = model_data['model']
    predicted_IPC_03 = model.predict(X_test_ts)
    predicted_speed_up = predicted_IPC_03 / X_test['IPC_19'].values[:, np.newaxis]
    model_data['predicted_speed_up'] = predicted_speed_up

# Calculate actual Speed_up
actual_speed_up = y_test.values / X_test['IPC_19'].values

# Plot Speed_up vs Cumulative Instructions for both models
plt.figure(figsize=(10, 6))
for model_name, model_data in models.items():
    predicted_speed_up_sorted = model_data['predicted_speed_up'][np.argsort(X_test['Cumulative Instructions'])]
    plt.plot(X_test_sorted['Cumulative Instructions'], predicted_speed_up_sorted, label=f'Predicted Speed_up ({model_name})')
plt.plot(X_test_sorted['Cumulative Instructions'], actual_speed_up[np.argsort(X_test['Cumulative Instructions'])], label='Actual Speed_up', color='skyblue', zorder=1)
plt.xlabel('Cumulative Instructions ($\\times10^{11}$)', fontsize=14, labelpad=20)
plt.ylabel('Speedup Factor', fontsize=14, labelpad=20)
plt.title('Predicted vs Actual Speedup Factor (LSTM vs GRU)', fontsize=16)
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e11:.0f}"))
plt.xlim(left=0)
# Set font size for ticks
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.show()
