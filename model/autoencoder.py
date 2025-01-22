import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Step 1: Load the dataset
# Replace "data.csv" with your dataset file path
# The dataset should contain sensor data (e.g., pressure, heart_rate)
data = pd.read_csv("synthetic_iot_anomaly_data.csv")  # pd.read_csv("data.csv")
print("Data shape:", data.shape)

# Step 2: Data Preprocessing
# Normalize the data to a range of 0 to 1
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Split into training and test sets
train_size = int(0.8 * len(data_scaled))
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

print("Training data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

# Step 3: Build the Autoencoder Model
input_dim = train_data.shape[1]

input_layer = Input(shape=(input_dim,))
encoder = Dense(64, activation="relu")(input_layer)
encoder = Dense(32, activation="relu")(encoder)
decoder = Dense(64, activation="relu")(encoder)
decoder = Dense(input_dim, activation="sigmoid")(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer="adam", loss="mse")

# Step 4: Train the Autoencoder
history = autoencoder.fit(
    train_data,
    train_data,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_split=0.2
)

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Step 5: Detect Anomalies
# Reconstruct the test data
reconstructed_data = autoencoder.predict(test_data)
reconstruction_errors = np.mean(np.square(test_data - reconstructed_data), axis=1)

# Define an anomaly threshold (95th percentile of errors)
threshold = np.percentile(reconstruction_errors, 95)

print(f"Anomaly threshold: {threshold}")

# Mark anomalies
anomalies = reconstruction_errors > threshold

# Step 6: Visualize Results
plt.hist(reconstruction_errors, bins=50)
plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2)
plt.title("Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.show()

# Print the number of anomalies
print(f"Number of anomalies detected: {np.sum(anomalies)}")

# Step 7: Evaluate (if you have true labels for anomalies)
# Replace y_true with actual anomaly labels if available
# y_true = [0, 0, 0, ..., 1, 1]  # Replace this with actual labels (0 for normal, 1 for anomaly)
# print(classification_report(y_true, anomalies))
