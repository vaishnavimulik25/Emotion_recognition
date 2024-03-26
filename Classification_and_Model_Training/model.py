import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import numpy as np

# Function to load features from a directory
def load_features_from_dir(directory):
    features = {}
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            filepath = os.path.join(directory, filename)
            feature = np.load(filepath)
            features[filename] = feature
    return features

# Load features and labels
def load_data(features_dir):
    loaded_features = load_features_from_dir(features_dir)
    labels = []
    for filename in loaded_features.keys():
        label = int(filename.split("_")[1][0])  # Extract label from filename
        labels.append(label)
    return list(loaded_features.values()), np.array(labels)

# Define CNN model
def create_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((input_shape[0], 1), input_shape=input_shape),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(256, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Define data directories
features_dir = "../data_preprocessing/extracted_features/Actor_01"

# Load data
features, labels = load_data(features_dir)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Get input shape and number of classes
input_shape = X_train[0].shape
num_classes = len(np.unique(labels))

# Create and compile the model
model = create_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Save the trained model
model.save("emotion_recognition_model.h5")
