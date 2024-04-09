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

# Define configurations of LSTM layers and dense layers
layer_configs = [
    {'lstm_layers': 1, 'dense_layers': 1},
    {'lstm_layers': 1, 'dense_layers': 2},
    {'lstm_layers': 2, 'dense_layers': 1},
    {'lstm_layers': 2, 'dense_layers': 2}
]

# Initialize lists to store results
results = {}

# Iterate over configurations
for config in layer_configs:
    lstm_layers = config['lstm_layers']
    dense_layers = config['dense_layers']

    # Initialize lists to store training set percentages and corresponding mean squared errors
    train_set_percentages = []
    mean_squared_errors = []

    # Iterate over each test set size
    for test_size in [0.2, 0.3, 0.4, 0.5]:
        # Calculate training set percentage
        train_percentage = (1 - test_size) * 100

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
        model = Sequential()
        for i in range(lstm_layers):
            if i == 0:
                model.add(LSTM(128, activation='relu', input_shape=(X_train_ts.shape[1], X_train_ts.shape[2]), return_sequences=True))
            else:
                model.add(LSTM(128, activation='relu', return_sequences=True))
        model.add(LSTM(128, activation='relu'))
        for _ in range(dense_layers):
            model.add(Dense(64, activation='relu'))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

        # Train the model
        history = model.fit(X_train_ts, y_train, epochs=100, batch_size=64, validation_split=0.1, verbose=0)

        # Evaluate the model
        mse = model.evaluate(X_train_ts, y_train)
        print(f"Mean Squared Error for training set percentage {train_percentage}%: {mse}")

        # Store training set percentage and mean squared error
        train_set_percentages.append(train_percentage)
        mean_squared_errors.append(mse)

    # Store results
    results[(lstm_layers, dense_layers)] = {'train_set_percentages': train_set_percentages, 'mean_squared_errors': mean_squared_errors}

# Plot results
for config, result in results.items():
    lstm_layers, dense_layers = config
    train_set_percentages = result['train_set_percentages']
    mean_squared_errors = result['mean_squared_errors']

    plt.plot(train_set_percentages, mean_squared_errors, marker='o', label=f'LSTM Layers: {lstm_layers}, Dense Layers: {dense_layers}')

plt.xlabel('Training Set Size (%)')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs Training Set Size for Different Model Configurations')
plt.legend()
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('mean_squared_error_train_set_percentage_plot.png')

# Show the plot
plt.show()
