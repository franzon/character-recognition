from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from time import time
import numpy as np
import pandas as pd
import time

import build_dataset


def classification():

    train_dataset = pd.read_csv('./dataset/Train.csv', delimiter=' ')
    test_dataset = pd.read_csv('./dataset/Test.csv', delimiter=' ')

    train_data = train_dataset.drop(columns=['label'])
    test_data = test_dataset.drop(columns=['label'])

    train_labels = train_dataset['label'].values
    test_labels = test_dataset['label'].values

    print('------ k-NN ------')

    t0 = time.time()
    k = 1
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_labels)
    print('k = {}'.format(k), knn.score(test_data, test_labels))
    print(time.time()-t0, 'segundos')

    t0 = time.time()
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_labels)
    print('k = {}'.format(k), knn.score(test_data, test_labels))
    print(time.time()-t0, 'segundos')

    t0 = time.time()
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_labels)
    print('k = {}'.format(k), knn.score(test_data, test_labels))
    print(time.time()-t0, 'segundos')

    print('------ SVM ------')

    t0 = time.time()
    kernel = 'linear'
    svm = SVC(kernel=kernel, gamma='auto')
    svm.fit(train_data, train_labels)
    print('kernel = {}'.format(kernel), svm.score(test_data, test_labels))
    print(time.time()-t0, 'segundos')

    t0 = time.time()
    kernel = 'linear'
    svm = SVC(kernel=kernel, C=0.5)
    svm.fit(train_data, train_labels)
    print('kernel = {} C=0.5'.format(kernel),
          svm.score(test_data, test_labels))
    print(time.time()-t0, 'segundos')

    t0 = time.time()
    kernel = 'linear'
    svm = SVC(kernel=kernel, C=0.1)
    svm.fit(train_data, train_labels)
    print('kernel = {} C=0.1'.format(kernel),
          svm.score(test_data, test_labels))
    print(time.time()-t0, 'segundos')

    print('------ Decision Tree ------')

    t0 = time.time()
    max_depth = 5
    dtree = DecisionTreeClassifier(max_depth=max_depth)
    dtree.fit(train_data, train_labels)
    print('max_depth = {}'.format(max_depth),
          dtree.score(test_data, test_labels))
    print(time.time()-t0, 'segundos')

    t0 = time.time()
    max_depth = 15
    dtree = DecisionTreeClassifier(max_depth=max_depth)
    dtree.fit(train_data, train_labels)
    print('max_depth = {}'.format(max_depth),
          dtree.score(test_data, test_labels))
    print(time.time()-t0, 'segundos')

    print('------ RandomForest ------')
    t0 = time.time()
    numberOfTrees = 120
    randomForest = RandomForestClassifier(
        n_estimators=numberOfTrees, min_samples_leaf=1)
    randomForest.fit(train_data, train_labels)
    print('Numero de Árvores = {}'.format(numberOfTrees),
          randomForest.score(test_data, test_labels))
    print(time.time()-t0, 'segundos')


sizes = [50]
pixels_per_cells = [10]

for size in sizes:
    for pixels_per_cell in pixels_per_cells:

        print('image_size = {}  pixels_per_cell = {} '.format(
            size, pixels_per_cell))

        build_dataset.build_dataset_hog_parametrized(
            size, pixels_per_cell, )
        classification()
        print('\n')
