#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# In[19]:


data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/website-data/master/dataset.csv")


# In[20]:


data.head()


# In[21]:


data.isnull().sum()


# In[22]:


data["language"].value_counts()


# In[23]:


data.count()


# In[24]:


x = np.array(data["Text"])
y = np.array(data["language"])

cv = CountVectorizer()
x = cv.fit_transform(x)
x_train , x_test ,y_train ,y_test = train_test_split(x,y,test_size=0.33 ,
                                                    random_state=42)


# In[25]:


model = MultinomialNB()
model.fit(x_train ,y_train)
model.score(x_test ,y_test)


# In[27]:


user = input("enter a text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)


# In[ ]:




