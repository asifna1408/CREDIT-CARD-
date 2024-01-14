#!/usr/bin/env python
# coding: utf-8

# In[71]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[ ]:





# In[8]:


df = pd.read_csv(r"C:\Users\asifn\OneDrive\Desktop\creditcard.csv (1)\creditcard.csv")


# In[34]:


df.head(50)


# In[10]:


df.shape


# In[14]:


df.size


# In[23]:


df.tail()


# In[24]:


# Datset information
df.info()


# In[26]:


# Checking the number of missing values in each column
df.isnull().sum()


# In[27]:


# if there is null value in the data, need to fill please refer


# In[28]:


#distribution of legit transaction and fraud transaction
df['Class'].value_counts()


# In[29]:


#This dataset is highly unbalanced O represent normal transactions and 1 represents fraud transaction


# In[31]:


#Seperating the data for analysis
legit= df[df.Class==0]
fraud= df[df.Class==1]


# In[32]:


print(legit.shape)
print(fraud.shape)


# In[33]:


#statistical measures of the data
legit.Amount.describe()


# In[35]:


fraud.Amount.describe()


# In[36]:


#compare the value for both transactions
df.groupby('Class').mean()


# In[37]:


# Under sampling
# Build a sample dataset containing similiar distribution of normal transactions and fradulent transactions
#Number of Fraudulent Transactions 492


# In[38]:


legit_sample = legit.sample(n=492)


# In[39]:


#concatenating two data frame
# if axis =0 it will represent changes in rows, if axis=1 , it will represent the changes in coloumns


# In[40]:


new_dataset = pd.concat([legit_sample,fraud], axis=0)


# In[41]:


new_dataset.head()


# ###### new_dataset.info()

# In[45]:


new_dataset['Class'].value_counts()


# In[46]:


new_dataset.tail()


# In[48]:


new_dataset.groupby('Class').mean()


# In[49]:


df.groupby('Class').mean()


# In[62]:


# splitting THE DATA IN TO FEATURES & TARGETS
X = new_dataset.drop(columns='Class',axis=1)
Y = new_dataset['Class']
print (X)
print (Y)


# In[65]:


#split the data in to Training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)


# In[72]:


#Check accuracy
#Logistic Regression
model = LogisticRegression()


# In[73]:


#Training Logistics Regression Model with Training Data
model.fit(X_train,Y_train)


# In[75]:


#Model Evaluation
#Accuracy Score
#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[80]:


print('Accuracy on Training data : ' , training_data_accuracy)


# In[81]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)


# In[82]:


print('Accuracy on Test data : ', test_data_accuracy)


# In[ ]:




