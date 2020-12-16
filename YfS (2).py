#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("Energy Consumption 3.csv")


# In[3]:


df.head() #There is a dependency of the conversion factor CO2 emissions.


# In[4]:


df.info()


# ## Determining the correlation between variables

# In[5]:


import seaborn as sns


# In[6]:


sns.pairplot(df)


# In[7]:


X=df[["CoveredBuildingArea","AreaAccordingToUse","TotalPopulation","EnergyConsumptionAverage4Years"]]
print(X)


# In[8]:


y=df["CO2Emissions"]
print(y)


# ## Splitting the model into training and test sets

# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)


# ## Fitting the data into our model

# In[11]:


from sklearn.linear_model import LinearRegression 


# In[12]:


lin=LinearRegression()


# In[13]:


lin.fit(X_train,y_train)
print("Training complete.")


# In[14]:


y_pred=lin.predict(X_test)


# In[15]:


print(y_pred)


# In[18]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 


#  ## Plotting the Multilinear Regression

# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[20]:


import statsmodels.formula.api as smf


# In[27]:


model = smf.ols(formula='CO2Emissions ~ EnergyConsumptionAverage4Years + AreaAccordingToUse', data=df)


# In[28]:


results_formula = model.fit()
results_formula.params


# In[29]:


import numpy as np


# In[30]:


x_surf, y_surf = np.meshgrid(np.linspace(df.EnergyConsumptionAverage4Years.min(), df.EnergyConsumptionAverage4Years.max(), 100),np.linspace(df.AreaAccordingToUse.min(), df.AreaAccordingToUse.max(), 100))


# In[31]:


onlyX = pd.DataFrame({'EnergyConsumptionAverage4Years': x_surf.ravel(), 'AreaAccordingToUse': y_surf.ravel()})


# In[32]:


fittedY=results_formula.predict(exog=onlyX)


# In[33]:


fittedY=np.array(fittedY)


# In[45]:


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(df['EnergyConsumptionAverage4Years'],df['AreaAccordingToUse'],df['CO2Emissions'],c='red', marker='o', alpha=0.5)
ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='b', alpha=0.3)
ax.set_xlabel('EnergyConsumptionAverage4Years')
ax.set_ylabel('AreaAccordingToUse')
ax.set_zlabel('CO2Emissions')
fig.set_size_inches(10,10)
plt.show()

