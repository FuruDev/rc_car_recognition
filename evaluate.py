import numpy as np
import tensorflow as tf
import keras
import matplotlib.image as mpimg
import pandas

nb_classes = 2

model_path = 'rc.h5'
model = keras.models.load_model(model_path)

# img_path = 'dataset/noncar/17.png'
# evaluate_set = [mpimg.imread(img_path)]

evaluate_df = pandas.read_csv('dataset/test.csv')
evaluate_set = []
# print(type(evaluate_set))
evaluate_class = evaluate_df['class']
evaluate_paths = evaluate_df['path']
#
for f in evaluate_paths:
    # print(f)
    evaluate_set.append(mpimg.imread(f))
#
evaluate_class = list(evaluate_class)

evaluate_set = np.asarray(evaluate_set)
evaluate_class = np.asarray(evaluate_class)

evaluate_class = keras.utils.to_categorical(evaluate_class, nb_classes)

evaluate_set = evaluate_set / 255.0

test_loss, test_acc = model.evaluate(evaluate_set, evaluate_class)
print('Test accuracy:', test_acc)