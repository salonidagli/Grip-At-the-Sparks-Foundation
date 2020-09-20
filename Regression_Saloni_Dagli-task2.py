#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv('http://bit.ly/w-data')


# In[3]:


data


# In[4]:


x=data.iloc[:,0].values
y=data.iloc[:,-1].values


# In[5]:


x


# In[6]:


y


# In[7]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)


# In[8]:


x_train=x_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)
y_test=y_test.reshape(-1,1)


# In[9]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[10]:


y_pred = regressor.predict(x_test)


# Visualising training set results

# In[11]:


plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Student Study Time')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# visualising test results

# In[12]:


plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

