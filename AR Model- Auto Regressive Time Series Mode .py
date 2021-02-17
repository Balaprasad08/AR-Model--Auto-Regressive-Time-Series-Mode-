#!/usr/bin/env python
# coding: utf-8

# ### import important library

# In[1]:


import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns


# ### Load Dataset

# In[2]:


os.chdir('E:\\prasad\\practice\\TimeSeries')


# In[3]:


df=pd.read_excel('ts_data.xlsx')


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.isnull().sum()


# ### Visualize Data is Stationary or Not

# In[7]:


plt.plot(df.Value)
plt.show()


# In[8]:


X=df.Value.values


# In[9]:


X.shape


# In[10]:


X


# ### Check Data is Stationary Or Not by using adfuller test

# In[11]:


import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose


# In[12]:


result=sts.adfuller(X)


# In[13]:


# p-value<0.5 So Reject the null hypothesis & accept the alternative hypothesis
# So Data is Stationary
result


# ### Create  ACF & PACF Plot

# In[14]:


sgt.plot_acf(X,lags=10)
plt.show()


# In[15]:


sgt.plot_pacf(X,lags=10)
plt.show()


# ### Data Shifted by 1

# In[16]:


df['values_shifted']=df['Value'].shift(1)


# In[17]:


df.head()


# In[18]:


df.drop('Time',axis=1,inplace=True)


# In[19]:


df.isnull().sum()


# In[20]:


df.dropna(inplace=True)


# In[21]:


df.isnull().sum()


# In[22]:


df.head()


# In[23]:


df.plot(figsize=(20,5))
plt.show()


# In[24]:


df.columns


# In[25]:


df.shape


# In[26]:


y=df.Value.values
X=df.values_shifted.values


# ### Split Data into Train & Test

# In[27]:


train_size=int(len(X)*0.80)


# In[28]:


X_train,X_test=X[0:train_size],X[train_size:len(X)]
y_train,y_test=y[0:train_size],y[train_size:len(X)]


# In[29]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[30]:


len(X_train)


# In[31]:


X_train.ndim


# In[32]:


X_test.ndim


# ### Convert X_train & X_test in 1D to 2D

# In[33]:


X_train=X_train.reshape(-1,1)
X_test=X_test.reshape(-1,1)


# In[34]:


X_train.ndim


# In[35]:


X_test.ndim


# ### Use Linear Regression Technique

# In[36]:


from sklearn.linear_model import LinearRegression


# In[37]:


lr=LinearRegression()


# In[38]:


lr.fit(X_train,y_train)


# In[39]:


y_pred=lr.predict(X_test)


# In[40]:


y_pred


# In[41]:


lr.coef_


# In[42]:


lr.intercept_


# ### Visualize the Actual & Predicted Values

# In[43]:


plt.plot(y_test[-10:],label='Actual',color='Blue')
plt.plot(y_pred[-10:],label='Predicted',color='Red')
plt.legend()
plt.show()


# ## Creat AR Model

# In[44]:


from statsmodels.tsa.arima_model import ARIMA


# In[45]:


model=ARIMA(y_train,order=(1,0,0))


# In[46]:


model_fit=model.fit()


# In[47]:


print(model_fit.summary())


# In[ ]:




