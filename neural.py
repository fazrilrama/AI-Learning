import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np

# Load dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisasi data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Bangun model neural network
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten layer untuk mengubah gambar menjadi vektor 1D
    Dense(128, activation='relu'),  # Hidden layer dengan 128 neuron dan aktivasi ReLU
    Dense(10, activation='softmax') # Output layer dengan 10 neuron (sesuai jumlah kelas) dan aktivasi softmax
])

# Compile model dengan pengoptimal dan fungsi loss
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predictions on test set
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# Plot a random sample of test images with predicted and true labels
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel(f'Predicted: {predicted_labels[i]} | True: {y_test[i]}')
plt.show()

# Evaluasi model dengan data uji
loss, accuracy = model.evaluate(x_test, y_test)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
