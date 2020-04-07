#!/usr/bin/env python
# coding: utf-8

# In[58]:


# Import libraries 

import pandas as pd
import numpy as np


# In[64]:


# Distribution of the target variable 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = 5,5
labels = hotel_booking['is_canceled'].value_counts().index.tolist()
sizes = hotel_booking['is_canceled'].value_counts().tolist()
explode = (0, 0.2)
colors = ['red','green']
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=False, startangle=30)
plt.axis('equal')
plt.tight_layout()
plt.title("Cancelation Dsitribution", fontdict=None, position= [0.48,1], size = 'xx-large')
plt.show() 


# In[65]:


# Trend of customers' stay on week nights month wise

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("dark")
sns.despine() 
plt.figure(figsize=(20,10))
plot = sns.lineplot(x="arrival_date_month", y="stays_in_week_nights", hue = 'hotel', palette = 'hot',
                  data=hotel_booking, ci =None )
plot.set_title('Average Stay in hotels on week nights per month',fontsize='xx-large') 
plot.set_ylabel('Avg. nights', fontsize = 15)  
plot.set_xlabel(' ') 


# In[66]:


# Trend of customers' stay on weekend nights month wise

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("dark")
sns.despine() 
plt.figure(figsize=(20,10))
plot = sns.lineplot(x="arrival_date_month", y="stays_in_weekend_nights", hue = 'hotel', palette = 'hot',
                  data=hotel_booking, ci= None )
plot.set_title('Average Stay in hotels on weekend nights per month',fontsize='xx-large') 
plot.set_ylabel('Avg. nights', fontsize = 15)  
plot.set_xlabel(' ') 


# In[67]:


# Distribution of type of customer 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

sns.catplot(x="customer_type", kind="count", palette="ch:.1000", data=hotel_booking);
plt.xlabel(' ',fontsize=12)
plt.ylabel('Frequency',fontsize=12)
plt.title('Customer Type Distribution',fontsize=15,fontweight ='bold') 


# In[68]:


# Distribution of Meal types 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(30,10))
sns.catplot(x="meal", kind="count", palette="ch:.85", data=hotel_booking);

plt.xlabel(' ',fontsize=12)
plt.ylabel('Frequency',fontsize=12)
plt.title('Meal Type Distribution',fontsize=15,fontweight ='bold') 


# In[69]:


# Distribution of city and resort hotel 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(30,10))
sns.catplot(x="hotel", kind="count", palette="ch:.85", data=hotel_booking);

plt.xlabel(' ',fontsize=12)
plt.ylabel('Frequency',fontsize=12)
plt.title('Hotel Distribution',fontsize=15,fontweight ='bold') 


# In[ ]:




