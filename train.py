import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas


train_df = pandas.read_csv('dataset/train.csv')
test_df = pandas.read_csv('dataset/test.csv')

train_paths = train_df['path']
test_paths = test_df['path']

train_class = train_df['class']
test_class = test_df['class']

car = []
noncar = []

for f in train_paths:
    # print(f)
    car.append(mpimg.imread(f))

for f in test_paths:
    # print(f)
    noncar.append(mpimg.imread(f))

class_names = ['car', 'noncar']

# train_paths = train_paths / 255.0
# test_paths = test_paths / 255.0

print(car)
print(noncar)

print(train_class)
print(test_class)




plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(car[i], cmap=plt.cm.binary)
    plt.xlabel(train_class[i])
#
model = keras.Sequential()
#
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
#
model.summary()
#
model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy', # ! sparse_categorical_crossentropy for without one-hot !
              metrics=['accuracy'])

model.fit(car[0], train_class, epochs=5)