# -*- coding: utf-8 -*-
"""Data_Analysis_Saloni_Dagli.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tygewvuTKrXdTJ-x2J2reLzLLergJybM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""Reading the data"""

data=pd.read_csv('SampleSuperstore.csv')

data.head()

data.shape

data.columns

data.nunique()

"""Describing the data"""

data.describe()

"""Sampling the data"""

data.sample(5)

"""Searching for null values if any"""

pd.isnull(data).values.any()

"""Data Visualisation"""

plt.hist(data['Discount'], bins = 10)
plt.xlabel('Discount')
plt.ylabel('Frequency')
plt.show()

plt.style.use('ggplot')
plt.hist(data['Profit'], bins = 10)
plt.xlabel('Profit')
plt.ylabel('Frequency')
plt.show()

"""Region-wise analysis"""

data['Region'].value_counts().plot(kind='bar')

data.groupby(data['Region'])['Profit'].mean().plot.bar()

fig = plt.figure(figsize =(50, 10)) 
plt.pie(data['Region'].value_counts(), labels = data['Region'].value_counts().index,autopct='%1.1f%%')
plt.title('Percentage of Deals in different Regions in USA')
plt.show()

"""We have found that the deals made are minimum in the South Region hence the company should do work on making more deals in the Southern Region

Sub-Category Analysis
"""

data['Sub-Category'].value_counts().plot(kind='bar')

data.groupby(data['Sub-Category'])['Sales'].max().plot.bar()

fig = plt.figure(figsize =(50, 20)) 
plt.pie(data['Sub-Category'].value_counts(), labels = data['Sub-Category'].value_counts().index,autopct='%1.1f%%')
plt.title('Percentage of Deals in different Sub-Categories in USA')
plt.show()

"""We have found that the minimum deals are made in copiers and machines hence the company should work on publicising these products more

State-wise Analysis
"""

data['State'].value_counts()

data['State'].value_counts().plot(kind='bar', figsize=(20,10),color='black')

fig = plt.figure(figsize =(50, 20)) 
plt.pie(data['State'].value_counts(), labels = data['State'].value_counts().index,autopct='%1.1f%%')
plt.title('Percentage of Deals in different States in USA')
plt.show()

"""There are multiple States like Wisconsin,Minnesota,Oklahama where the deals made are close to none hence the company should focus on making more deals in these States

City-wise Analysis
"""

data['City'].value_counts()

data['City'].value_counts().head(100).plot(kind='bar', figsize=(20,10),color='blue')

fig = plt.figure(figsize =(50, 10)) 
plt.pie(data['City'].value_counts(), labels = data['City'].value_counts().index,autopct='%1.1f%%')
plt.title('Percentage of Deals in different cities in USA')
plt.show()

"""There are multiple Cities where the deals made are close to none hence the company should focus on making more deals in these Cities.

Bivariate Analysis
"""

plt.scatter(data['Discount'],data['Sales'])

plt.scatter(data['Discount'],data['Profit'])

sns.pairplot(data)

plt.figure(figsize=(8,8))
sns.heatmap(data.corr(),annot=True,square=True,cmap='viridis')
plt.yticks(rotation=0)

sns.distplot(data['Profit'],color='black')