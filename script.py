from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import numpy as np
import pandas as pd
import time

import build_dataset


def classification(arq):

    train_dataset = pd.read_csv('./dataset/Train.csv', delimiter=' ')
    test_dataset = pd.read_csv('./dataset/Test.csv', delimiter=' ')

    train_data = train_dataset.drop(columns=['label'])
    test_data = test_dataset.drop(columns=['label'])

    train_labels = train_dataset['label'].values
    test_labels = test_dataset['label'].values

    print('------ k-NN ------')

    k = 1
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_labels)
    print('k = {}'.format(k), knn.score(test_data, test_labels))

    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_labels)
    print('k = {}'.format(k), knn.score(test_data, test_labels))
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_labels)
    print('k = {}'.format(k), knn.score(test_data, test_labels))

    print('------ SVM ------')

    kernel = 'linear'
    svm = SVC(kernel=kernel, gamma='auto')
    svm.fit(train_data, train_labels)
    print('kernel = {}'.format(kernel), svm.score(test_data, test_labels))

    kernel = 'linear'
    svm = SVC(kernel=kernel, C=0.5)
    svm.fit(train_data, train_labels)
    print('kernel = {} C=0.5'.format(kernel),
          svm.score(test_data, test_labels))

    kernel = 'linear'
    svm = SVC(kernel=kernel, C=0.25)
    svm.fit(train_data, train_labels)
    print('kernel = {} C=0.25'.format(kernel),
          svm.score(test_data, test_labels))

    kernel = 'linear'
    svm = SVC(kernel=kernel, C=0.1)
    svm.fit(train_data, train_labels)
    print('kernel = {} C=0.1'.format(kernel),
          svm.score(test_data, test_labels))

    kernel = 'rbf'
    svm = SVC(kernel=kernel, gamma='auto')
    svm.fit(train_data, train_labels)
    print('kernel = {}'.format(kernel), svm.score(test_data, test_labels))

    kernel = 'sigmoid'
    svm = SVC(kernel=kernel, gamma='auto')
    svm.fit(train_data, train_labels)
    print('kernel = {}'.format(kernel), svm.score(test_data, test_labels))

    print('------ Decision Tree ------')

    max_depth = 5
    dtree = DecisionTreeClassifier(max_depth=max_depth)
    dtree.fit(train_data, train_labels)
    print('max_depth = {}'.format(max_depth),
          dtree.score(test_data, test_labels))

    max_depth = 10
    dtree = DecisionTreeClassifier(max_depth=max_depth)
    dtree.fit(train_data, train_labels)
    print('max_depth = {}'.format(max_depth),
          dtree.score(test_data, test_labels))

    max_depth = 15
    dtree = DecisionTreeClassifier(max_depth=max_depth)
    dtree.fit(train_data, train_labels)
    print('max_depth = {}'.format(max_depth),
          dtree.score(test_data, test_labels))


filter_ = [False, True]
sizes = [50, 100]
orientations = [4, 8]
pixels_per_cells = [5, 10, 25]

for f in filter_:
    for size in sizes:
        for orientation in orientations:
            for pixels_per_cell in pixels_per_cells:
                arq = open('{}x{}-{}-{}-{}-{}.txt'.format(size, size,
                                                          orientation, pixels_per_cell, pixels_per_cell, f),)
                print('image_size = {}x{} orientations = {} pixels_per_cell = {}x{} filter = {}'.format(
                    size, size, orientation, pixels_per_cell, pixels_per_cell, f))

                build_dataset.build_dataset_hog_parametrized(
                    orientation, size, pixels_per_cell, filter_)
                classification(arq)
                print('\n')
