import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas

height = 128
width = 128
nb_channels = 3
nb_classes = 2

train_df = pandas.read_csv('dataset/train.csv')
test_df = pandas.read_csv('dataset/test.csv')

train_paths = train_df['path']
test_paths = test_df['path']

train_class = train_df['class']
test_class = test_df['class']

train_set = []
test_set = []

for f in train_paths:
    # print(f)
    train_set.append(mpimg.imread(f))

for f in test_paths:
    # print(f)
    test_set.append(mpimg.imread(f))

class_names = ['car', 'noncar']

# print(train_set)
# print(test_set)

# print(train_class)
# print(test_class)

train_set = np.asarray(train_set)
test_set = np.asarray(test_set)

train_class = list(train_class)
test_class = list(test_class)

# print(train_class)

train_class = np.asarray(train_class)
test_class = np.asarray(test_class)

train_class = keras.utils.to_categorical(train_class, nb_classes)
test_class = keras.utils.to_categorical(test_class, nb_classes)

train_set = train_set / 255.0
test_set = test_set / 255.0

# print(train_set.shape)
# print(train_class.shape)

# print(train_set.shape)
model = keras.Sequential()
#
# model.add(keras.layers.Flatten(input_shape=(128, 128, 3)))
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dense(1, activation='softmax'))

model.add(keras.layers.Conv2D(32,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding='same',
                              input_shape=(height, width, nb_channels)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.BatchNormalization(axis=-1))
model.add(keras.layers.Conv2D(32,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding='same'))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.BatchNormalization(axis=-1))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(64,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding='same'))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.BatchNormalization(axis=-1))
model.add(keras.layers.Conv2D(64,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding='same'))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.BatchNormalization(axis=-1))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(nb_classes))
model.add(keras.layers.Activation('sigmoid')) # ! Activation cannot be softmax
# solved: https://stackoverflow.com/questions/45378493/why-does-a-binary-keras-cnn-always-predict-1

model.summary()

model.compile(optimizer='rmsprop', # optimizer change from Adam to rmsprop
              loss='binary_crossentropy', # ! sparse_categorical_crossentropy for without one-hot !
              metrics=['accuracy'])

model.fit(train_set, train_class, epochs=5)

model.save('rc.h5')