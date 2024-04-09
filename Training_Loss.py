import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Loading dataset
dataset = pd.read_csv("Merged_IPC_6.csv")

# Splitting dataset into features (X) and target variable (y)
X = dataset[['Cumulative Instructions', 'IPC_19']]
y = dataset['IPC_03']

# Defining test set size
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
history = model.fit(X_train_ts, y_train, epochs=100, batch_size=64, validation_split=0.1, verbose=1)

# Capture training loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot training loss
fig, ax = plt.subplots()

def animate(i):
    ax.clear()
    ax.plot(range(len(train_loss[:i])), train_loss[:i], label='Training Loss')
    ax.plot(range(len(val_loss[:i])), val_loss[:i], label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title('Training and Validation Loss')

ani = animation.FuncAnimation(fig, animate, frames=len(train_loss), interval=100, repeat=False)

# Save GIF
ani.save('training_progress.gif', writer='pillow')
