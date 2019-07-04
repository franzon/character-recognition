from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import time

train_dataset = pd.read_csv('./dataset/Train.csv', delimiter=' ')
test_dataset = pd.read_csv('./dataset/Test.csv', delimiter=' ')

train_data = train_dataset.drop(columns=['label'])
test_data = test_dataset.drop(columns=['label'])

train_labels = train_dataset['label'].values
test_labels = test_dataset['label'].values

print('------ k-NN ------')
labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

k = 1
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(train_data, train_labels)
knn_predict = knn.predict(test_data)
print(knn.score(test_data, test_labels))


cm_train = confusion_matrix(test_labels, knn_predict)
df_cm = pd.DataFrame(cm_train, index=[i for i in labels],
                     columns=[i for i in labels])

plt.figure(figsize=(20, 20))
sn.heatmap(df_cm, annot=True)
plt.show()
