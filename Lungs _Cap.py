#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd


# In[3]:


# Linear Regression
# steps to be followed

#data import
#data cleaning-->null,replace
#Sampling-->train_test_splite
#Linreg
#Build Model-->fit
#linreg.score-->Rsquare
#AdjRsquare
#linreg.intercept_
#linreg.coef_
#pred-->train_x,test_x
#err_train
#err_mean()
#hist plot
#skew()
#kurt+3
#scatter plot
#data fram-->x=Actual y=pred
#regplot
#err_test
#mse
#rmse
#mape
#acc
# To be taken as note
# 1) data should be numeric and continuous only
# 2) hist should be normaly distributed
# 3) Data should be equaly scattered
# 4) in regplot all data should be linear pred line


# In[4]:


import pandas as pd


# In[5]:


# LungCapData


# In[6]:


pd.read_csv(r"C:\Users\user\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\F81493623DA4A6C582DE384F23169A366E91AA5E\transfers\2026-37\LungCapData.csv")


# In[7]:


lung=pd.read_csv(r"C:\Users\user\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\F81493623DA4A6C582DE384F23169A366E91AA5E\transfers\2026-37\LungCapData.csv")


# In[8]:


lung.isnull().sum()[lung.isnull().sum()>0]


# In[9]:


lung.shape


# In[10]:


lung.Smoke.replace({'no':0,'yes':1},inplace=True)
lung.Gender.replace({'male':1,'female':0},inplace=True)
lung.Caesarean.replace({'no':0,'yes':1},inplace=True)


# In[11]:


lung.head()


# In[12]:


from sklearn.model_selection import train_test_split
lung_train,lung_test=train_test_split(lung,test_size=.2)


# In[13]:


lung_train_x=lung_train.iloc[::,1:]
lung_train_y=lung_train.iloc[::,0]


# In[14]:


lung_test_x=lung_test.iloc[::,1:]
lung_test_y=lung_test.iloc[::,0]


# In[15]:


lung_test_x


# In[16]:


from sklearn.linear_model import LinearRegression
linreg=LinearRegression()


# In[17]:


linreg.fit(lung_train_x,lung_train_y)


# In[18]:


linreg.score(lung_train_x,lung_train_y)


# In[19]:


lung_train_x.shape[0]


# In[20]:


lung_train_x.shape[1]


# In[21]:


lung_train_x.shape


# In[22]:


Rsquare=linreg.score(lung_train_x,lung_train_y)
n=lung_train_x.shape[0]
k=lung_train_x.shape[1]
adjrsquare=1-(1-Rsquare)*(n-1)/(n-k-1)
adjrsquare


# In[23]:


print(linreg.intercept_)


# In[24]:


linreg.coef_


# In[25]:


pred_train=linreg.predict(lung_train_x)
pred_test=linreg.predict(lung_test_x)


# In[26]:


err_train=lung_train_y-pred_train


# In[27]:


err_train


# In[28]:


err_train.mean()


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[30]:


plt.hist(err_train,edgecolor='red',bins=20);plt.hist(err_train,edgecolor='red',bins=20);


# In[31]:


err_train.skew()


# In[32]:


err_train.kurtosis()+3


# In[33]:


plt.plot(err_train,'*')


# In[34]:


pred_act=pd.DataFrame()
pred_act['Actual']=lung_train_y
pred_act['Pred']=pred_train
pred_act


# In[35]:


sns.regplot(x='Actual',y='Pred',data=pred_act)


# In[36]:


err_test=lung_test_y-pred_test
err_test


# In[37]:


import numpy as np


# In[38]:


mse=np.mean(np.square(err_test))
mse


# In[39]:


np.sqrt(mse)


# In[40]:


mape=np.mean(np.abs(err_test*100/lung_test_y))


# In[41]:


mape


# In[42]:


acc=100-mape
print(acc)


# In[43]:


get_ipython().system('pip install streamlit')


# In[44]:


# model saved


# In[45]:


import pickle

# 'linreg' is the exact name of your trained model from your screenshot
with open('lung_cap_model.pkl', 'wb') as file:
    pickle.dump(linreg, file)

print("Success! Your model has been saved as 'lung_cap_model")


# In[ ]:




