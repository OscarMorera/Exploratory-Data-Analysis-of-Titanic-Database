#!/usr/bin/env python
# coding: utf-8

# ### Exploratory Data Analysis of Titanic Database that we can find on https://www.kaggle.com/c/titanic/data

# In[83]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# In[84]:


passenger_data = pd.read_csv('titanic.csv')
passenger_data.head(10)


# In[85]:


passenger_data.describe()


# In[86]:


survived_data = passenger_data[passenger_data['Survived'] == 1]
survived = survived_data.count().values[1]
survival_percent = (survived/891) * 100
print('Survivors in training data are {}'.format(survival_percent))


# In[87]:


survival_rate = passenger_data[['Pclass', 'Sex','Survived']].groupby(['Pclass', 'Sex'], as_index = False).mean().sort_values('Survived', ascending = False)
print(survival_rate)


# In[88]:


Cols = ['Pclass', 'Sex','Survived']
for index in Cols:
     passenger_data[index] = passenger_data[index].astype(str)
passenger_data.dtypes


# In[89]:


passenger_data.loc[passenger_data.duplicated(), :]


# In[90]:


train=passenger_data
test=passenger_data
train.isnull().sum


# In[91]:


test.isnull().sum


# In[92]:


impute_value = train['Age'].median()
train['Age'] = train['Age'].fillna(impute_value)
test['Age'] = test['Age'].fillna(impute_value)


# In[93]:


train['IsFemale'] = (train['Sex'] == 'female').astype(int)
test['IsFemale'] = (test['Sex'] == 'female').astype(int)


# In[94]:


predictors = ['Pclass', 'IsFemale', 'Age']
X_train = train[predictors].values
X_test = test[predictors].values
y_train = train['Survived'].values


# In[95]:


X_train[:5]


# In[96]:


y_train[:5]


# In[97]:


from sklearn.linear_model import LogisticRegression


# In[98]:


model = LogisticRegression()


# In[99]:


model.fit(X_train, y_train)


# In[100]:


y_predict = model.predict(X_test)
y_predict[:5]

