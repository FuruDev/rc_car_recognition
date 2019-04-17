import csv
from os import listdir
from os.path import isfile, join
import sklearn.model_selection
import pandas


f = []

car_path = 'dataset/car/'
car = [f for f in listdir(car_path) if isfile(join(car_path, f))]
noncar_path = 'dataset/noncar/'
noncar = [f for f in listdir(noncar_path) if isfile(join(noncar_path, f))]


for file in car:
    f.append((car_path + file, 1))

for file in noncar:
    f.append((noncar_path + file, 0))

with open('dataset.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(('path', 'class'))
    writer.writerows(f)

dataframe = pandas.read_csv('dataset.csv')

a, b = sklearn.model_selection.train_test_split(dataframe, random_state=10)

a.to_csv('dataset/train.csv', index=False)
b.to_csv('dataset/test.csv', index=False)