#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px


# In[2]:


workface = pd.read_csv('deliverytime.txt')


# In[3]:


workface.head()


# In[4]:


workface.info()


# In[19]:


workface.describe()


# In[5]:


workface.isnull().sum()


# In[9]:


R = 6371

def deg_to_rad(degrees):
    return degrees*(np.pi/180)

def distcalculate(lat1,lon1,lat2,lon2):
    d_lat = deg_to_rad(lat2-lat1)
    d_lon = deg_to_rad(lon2-lon1)
    a=np.sin(d_lat/2)**2 +np.cos(deg_to_rad(lat1))*np.cos(deg_to_rad(lat2))*np.sin(d_lon/2)**2
    c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    return R*c
for i in range(len(workface)):
    workface.loc[i,'distance'] = distcalculate(workface.loc[i,'Restaurant_latitude'],
                                          workface.loc[i,'Restaurant_longitude'],
                                          workface.loc[i,'Delivery_location_latitude'],
                                          workface.loc[i,'Delivery_location_longitude'])


# In[10]:


workface.head()


# In[12]:


figure = px.scatter(data_frame=workface,
                    x="distance",
                    y="Time_taken(min)",
                    size="Time_taken(min)",
                    trendline="ols",  # Corrected the typo here
                    title="Relationship between distance and time taken")
figure.show()


# In[13]:


figure = px.scatter(data_frame=workface,
                    x="Delivery_person_Age",
                    y="Time_taken(min)",
                    size="Time_taken(min)",
                    color="distance",
                    trendline="ols",  # Corrected the typo here
                    title="Relationship between time taken and age")
figure.show()


# In[14]:


figure = px.scatter(data_frame=workface,
                    x="Delivery_person_Ratings",
                    y="Time_taken(min)",
                    size="Time_taken(min)",
                    color="distance",
                    trendline="ols",  # Corrected the typo here
                    title="Relationship between time taken and Ratings")
figure.show()


# In[15]:


fig = px.box(workface,
            x="Type_of_vehicle",
            y="Time_taken(min)",
            color="Type_of_order")
fig.show()


# In[16]:


from sklearn.model_selection import train_test_split
x = np.array(workface[["Delivery_person_Age",
                   "Delivery_person_Ratings",
                   "distance"]])
y = np.array(workface[["Time_taken(min)"]])
xtrain,xtest,ytrain,ytest =train_test_split(x,y,
                                           test_size=0.10,
                                           random_state=42)
from keras.models import Sequential
from keras.layers import Dense,LSTM
model = Sequential()
model.add(LSTM(128,return_sequences=True,input_shape=(xtrain.shape[1],1)))
model.add(LSTM(64,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()


# In[17]:


model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(xtrain,ytrain,batch_size=1,epochs=9)


# In[18]:


print("Food Delivery Time prediction")
a = int(input("Age of Delivery partner: "))
b = float(input("Ratings of previous Deliveries: "))
c = int(input("Total Distance: "))

features = np.array([[a,b,c]])
print("Predicted Delivery Time in Minutes = ",model.predict(features))


# In[ ]:




