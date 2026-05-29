# CNN Mini Project - Image Classification using CNN
# Dataset: CIFAR-10
# Author: Your Name

# Import Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Module 1: Data Collection
# -------------------------------

# Load CIFAR-10 Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Class Labels
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat',
               'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

print("Training Images Shape:", x_train.shape)
print("Testing Images Shape:", x_test.shape)

# -------------------------------
# Module 2: Image Preprocessing
# -------------------------------

# Normalize images (0 to 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Display Sample Images
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[np.argmax(y_train[i])])
    plt.axis('off')

plt.show()

# -------------------------------
# Module 3: CNN Model Building
# -------------------------------

model = Sequential()

# First Convolution Layer
model.add(Conv2D(32, (3, 3), activation='relu',
                 input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))

# Second Convolution Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Third Convolution Layer
model.add(Conv2D(64, (3, 3), activation='relu'))

# Flatten Layer
model.add(Flatten())

# Dense Layers
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(10, activation='softmax'))

# Model Summary
model.summary()

# -------------------------------
# Module 4: Model Training
# -------------------------------

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# -------------------------------
# Module 5: Prediction
# -------------------------------

# Predict on test images
predictions = model.predict(x_test)

# Display Predictions
plt.figure(figsize=(10, 10))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[i])

    predicted_label = class_names[np.argmax(predictions[i])]
    actual_label = class_names[np.argmax(y_test[i])]

    plt.title(f"Pred: {predicted_label}\nActual: {actual_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# -------------------------------
# Module 6: Evaluation
# -------------------------------

# Evaluate Model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

# -------------------------------
# Accuracy Graph
# -------------------------------

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Accuracy Graph')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# -------------------------------
# Loss Graph
# -------------------------------

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Loss Graph')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# -------------------------------
# Save Model
# -------------------------------

model.save("cnn_cifar10_model.h5")

print("\nModel saved successfully!")