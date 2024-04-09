import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Loading dataset
dataset = pd.read_csv("Merged_IPC_6.csv")

# Splitting dataset into features (X) and target variable (y)
X = dataset[['Cumulative Instructions', 'IPC_19']]
y = dataset['IPC_03']

# Defining a range of test set sizes
test_set_sizes = [0.2, 0.3, 0.4, 0.5]

# Initializing lists to store test set sizes and corresponding mean squared errors
test_sizes = []
mean_squared_errors = []

# Iterating over each test set size
for test_size in test_set_sizes:
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

    # Store test set size and mean squared error
    test_sizes.append(test_size)
    mean_squared_errors.append(mse)

    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch', fontsize=14, labelpad=20)
    plt.ylabel('Loss', fontsize=14, labelpad=20)
    plt.legend()
    plt.title(f'Training and Validation Loss', fontsize=14)
    # Set the x-axis limits to start from 0
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    # Set font size for ticks
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.grid(True)
    plt.show()

    # Plot predicted vs actual values
    predicted_IPC_03 = model.predict(X_test_ts)

    # Plot IPC predictions vs Cumulative Instructions
    plt.figure(figsize=(10, 6))

    # Sort the data by Cumulative Instructions for better visualization
    sorted_indices = X_test['Cumulative Instructions'].argsort()
    X_test_sorted = X_test.iloc[sorted_indices]
    predicted_IPC_03_sorted = predicted_IPC_03[sorted_indices]
    y_test_sorted = y_test.values[sorted_indices]

    plt.plot(X_test_sorted['Cumulative Instructions'], y_test_sorted, label='Actual IPC P(big) Core', color='purple')
    plt.plot(X_test_sorted['Cumulative Instructions'], predicted_IPC_03_sorted, label='Predicted IPC P(big) Core', color='red')
    plt.xlabel('Cumulative Instructions ($\\times10^{11}$)', fontsize=14, labelpad=20)
    plt.ylabel('Instructions per Cycle (IPC)', fontsize=14, labelpad=20)
    plt.title(f'IPC Predictions vs Cumulative Instructions', fontsize=16)
    plt.legend()
    plt.grid(True)


    # Format x-axis ticks as scientific notation with 10^11
    def format_ticks(x, _):
        if int(x / 1e11) == 0:
            return f'{int(x)}'
        else:
            return f'{int(x / 1e11)}x10^11'


    # Set the x-axis limits to start from 0
    plt.xlim(left=0)
    # Set font size for ticks
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)

    # Apply scientific notation formatting to x-axis ticks
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1e11:.0f}"))
    plt.show()

    # Calculate predicted Speed_up
    predicted_Speed_up = predicted_IPC_03 / X_test['IPC_19'].values[:, np.newaxis]

    # Sort the data by Cumulative Instructions for better visualization
    sorted_indices = X_test['Cumulative Instructions'].argsort()
    X_test_sorted = X_test.iloc[sorted_indices]
    predicted_Speed_up_sorted = predicted_Speed_up[sorted_indices]
    actual_Speed_up_sorted = dataset['Speed_up'].iloc[X_test_sorted.index]

    # Plot actual and predicted Speed_up
    plt.figure(figsize=(12, 8))

    # Sort the data by Cumulative Instructions for better visualization
    sorted_indices = X_test['Cumulative Instructions'].argsort()
    X_test_sorted = X_test.iloc[sorted_indices]
    predicted_Speed_up_sorted = predicted_Speed_up[sorted_indices]
    actual_Speed_up_sorted = dataset['Speed_up'].iloc[X_test_sorted.index]

    plt.plot(X_test_sorted['Cumulative Instructions'], actual_Speed_up_sorted, label='Actual Speed_up', color='skyblue')
    plt.plot(X_test_sorted['Cumulative Instructions'], predicted_Speed_up_sorted, label='Predicted Speed_up', color='red')
    plt.xlabel('Cumulative Instructions ($\\times10^{11}$)', fontsize=14, labelpad=20)
    plt.ylabel('Speedup', fontsize=14, labelpad=20)
    plt.title(f'Actual vs Predicted Speedup', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    # Format x-axis ticks as scientific notation with 10^11
    def format_ticks(x, _):
        if int(x / 1e11) == 0:
            return f'{int(x)}'
        else:
            return f'{int(x / 1e11)}x10^11'


    # Set the x-axis limits to start from 0
    plt.xlim(left=0)
    # Set font size for ticks
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)

    # Apply scientific notation formatting to x-axis ticks
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1e11:.0f}"))
    plt.show()

# Plot errors against the size of the test set
plt.plot(test_sizes, mean_squared_errors, marker='o')
plt.xlabel('Test Set Size')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs Test Set Size')
plt.grid(True)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x*100)}%"))
plt.show()


