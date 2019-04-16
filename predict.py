import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas


model_path = 'rc.h5'
model = keras.models.load_model(model_path)

img_path = 'dataset/noncar/15.png'
predict_set = [mpimg.imread(img_path)]

predict_set = np.asarray(predict_set)
# print(predict_set.shape)

predict_set = predict_set / 255.0
# print(predict_set)

prediction = model.predict(predict_set)
print(prediction)
_pred = prediction.argmax()
print(_pred)