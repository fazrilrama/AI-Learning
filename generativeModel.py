import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras.models import Model
import numpy as np

# Definisi fungsi TensorFlow yang ingin Anda gunakan dalam Keras
def tf_fn(x):
    return tf.square(x)  # Mengkuadratkan input

# Definisi custom layer Keras untuk memanfaatkan tf_fn
class MyLayer(Layer):
    def call(self, inputs):
        return tf_fn(inputs)

# Contoh penggunaan dalam model Keras
inputs = Input(shape=(1,))
x = MyLayer()(inputs)  # Memanggil MyLayer dengan input 'inputs'
outputs = Dense(1)(x)  # Contoh layer Dense setelah MyLayer

# Membuat model Keras
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')  # Compile model dengan optimizer dan loss function

# Melakukan training dengan data dummy
x_train = np.array([[1.0], [2.0], [3.0]])
y_train = np.array([[1.0], [4.0], [9.0]])
model.fit(x_train, y_train, epochs=50, verbose=1)

# Evaluasi model
x_test = np.array([[4.0], [5.0]])
y_pred = model.predict(x_test)
print("Predictions:", y_pred)
