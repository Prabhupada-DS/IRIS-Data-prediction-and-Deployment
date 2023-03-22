#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[4]:


df=pd.read_csv('D:\\Dataset\Iris.csv')
df.head()


# In[5]:


# delete a column
df = df.drop(columns = ['Id'])
df.head()


# In[8]:


x = df.drop(columns=['Species'])
y = df['Species']


# In[9]:


# Split the data to train and test dataset.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[10]:


# Support vector machine algorithm
from sklearn.svm import SVC
svn = SVC()
svn.fit(x_train, y_train)


# In[11]:


# Predict from the test dataset
predictions = svn.predict(x_test)
# Calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


# In[12]:


# A detailed classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[13]:


# Testing the model
x_new = np.array([[3, 2, 1, 0.2], [  4.9, 2.2, 3.8, 1.1 ], [  5.3, 2.5, 4.6, 1.9 ]])
#Prediction of the species from the input vector
prediction = svn.predict(x_new)
print("Prediction of Species: {}".format(prediction))


# In[14]:


import pickle


# In[15]:


filename='iris.pkl'
pickle.dump(svn,open(filename,'wb'))


# In[17]:


load_model = pickle.load(open(filename,'rb'))


# In[18]:


load_model.predict([[6.0, 2.2, 4.0, 1.0]])


# In[ ]:




