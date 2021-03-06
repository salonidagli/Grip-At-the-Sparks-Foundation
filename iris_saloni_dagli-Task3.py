# -*- coding: utf-8 -*-
"""iris_saloni_dagli.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EHss2xNxQN7CrskmXtB9GhyxfItzz7GJ
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import seaborn as sns

x = pd.read_csv('Iris.csv')

"""Dataset imported"""

x

x.drop(['Id'],axis=1,inplace=True)

x.head()

"""Checking for null values"""

x.isnull().values.any()

"""No null values

Data Visualisation
"""

sns.pairplot(x)

plt.figure(figsize=(8,8))
sns.heatmap(x.corr(),annot=True,square=True,cmap='viridis')
plt.yticks(rotation=0)

from sklearn import preprocessing 
  
label_encoder = preprocessing.LabelEncoder() 

x['Species']= label_encoder.fit_transform(x['Species'])

"""Label encoding performed on the categorical data"""

x.head()

x = x.iloc[:, [0, 1, 2, 3]].values

x

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

"""The elbow is at 3 hence the optimal no. of clusters are 3"""

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

y_kmeans

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

plt.legend()