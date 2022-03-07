import numpy as np
from tensorflow.keras import layers
import tensorflow as tf

@tf.custom_gradient
def custom_floor(x):
    def grad_fn(dy):
        return dy

    return tf.floor(x), grad_fn

def fp_Quantize(x, w, f):
    i = w - f
    max = float(2 ** (i - 1) - 2 ** (-f))
    min = float(-2 ** (i - 1))
    n = float(2 ** f)
    xx = custom_floor(x * n + 0.5) / n
    clipped = tf.keras.backend.clip(xx, min_value=min, max_value=max)
    return clipped

class Dense_FP(tf.keras.layers.Layer):
    def __init__(self, units=32, Length=6,Fr=3):
        super(Dense_FP, self).__init__()
        self.units = units
        self.length = Length
        self.fr = Fr
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)
        self.gamma = self.add_weight(shape=(1,), initializer="ones", trainable=True)
         
    def call(self, inputs):
        xq = fp_Quantize(inputs, self.length, 0)*self.gamma
        wq = fp_Quantize(self.w, self.length, 0)*self.gamma
        bq = fp_Quantize(self.b, self.length, 0)*self.gamma
        return tf.matmul(xq, wq) + bq


class Conv2D_FP(tf.keras.layers.Layer):
    def __init__(self, nfilters=32, k=3, padding='same', strides=(1, 1),act='relu', Length=6,Fr=3, **kwargs):
        super(Conv2D_FP, self).__init__(**kwargs)
        self.nfilters = nfilters
        self.k = k
        self.padding = padding
        self.strides = strides
        self.kernel_initializer = 'glorot_uniform'
        self.length = Length
        self.fr = Fr
        self.act = act

    def build(self, input_shape):

        self.w = self.add_weight(shape=(self.k,self.k,input_shape[-1],self.nfilters),
                                    name='kernel',
                                    trainable= True,
                                    initializer=self.kernel_initializer)
        self.b = self.add_weight(shape=(self.nfilters,),name='bias',trainable= True,initializer="zeros")

    def call(self, x, mask=None):
        xq = fp_Quantize(x,self.length,self.fr)
        wq = fp_Quantize(self.w,self.length,self.fr)
        bq = fp_Quantize(self.b,self.length,self.fr)

        return tf.keras.backend.conv2d(xq, kernel=wq, padding=self.padding, strides=self.strides) + bq


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 256
x_test = x_test.astype("float32") / 256
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

L = 8
F =4
model = tf.keras.Sequential()
model.add(layers.InputLayer(input_shape=input_shape))
#model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D_FP(nfilters=32, k=3, padding='valid',Length=L,Fr=F,act="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D_FP(nfilters=64, k=3, padding='valid',Length=L,Fr=F,act="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
#model.add(layers.Dense(num_classes, activation="softmax"))
model.add(Dense_FP(num_classes,Length=L,Fr=F))
model.add(layers.Softmax())
model.summary()
batch_size = 256
epochs = 20

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
