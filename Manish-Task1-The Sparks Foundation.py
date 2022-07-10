#!/usr/bin/env python
# coding: utf-8

# In[10]:


print("The Sparks Foundation")
print("Data Science and Business Analytics")
print("Task1: Prediction using supervised ML")
print("Author: Manish Kumawat")


# In[1]:


import http
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
url = "http://bit.ly//w-data"
data=pd.read_csv(url)
data.head(10)


# In[2]:


data.describe()


# In[3]:


data.isnull().sum()


# In[4]:


data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours studied vs Marks scored')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Score')
plt.show


# In[5]:


x=data.iloc[:, :-1]
y=data.iloc[:, 1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)


# In[6]:


y_pred=model.predict(x_test)
df=pd.DataFrame({'ACTUAL_VALUE' :y_test, 'PREDICT_VALUE':y_pred})
df


# In[7]:


print('Prdicted Score is' ,model.predict([[9.25]]))


# In[9]:


from sklearn import metrics
print('Mean Absolute Error:',
       metrics .mean_absolute_error(y_test, y_pred))


# In[ ]:




