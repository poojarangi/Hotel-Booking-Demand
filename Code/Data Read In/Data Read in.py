#!/usr/bin/env python
# coding: utf-8

# ## Hotel Booking Demand  
# Created by: Pooja Rangi
# 
# The dataset Hotel_Bookings from the article Hotel Booking Demand Datasets, written by Nuno Antonio, Ana Almeida, and Luis Nunes for Data in Brief, Volume 22, February 2019.
# 
# It contains hotel booking information for a city and a resort hotel that includes information such as when the booking was made, length of stay, the number of adults, children, and/or babies, and the number of available parking spaces, among other things.
# 
# Note: No personnel information is showcased in the data.
# 
# #### Analyses:
# The first part contains descriptive stats and EDA. 
# The second part contains a classification model to find out if a customer will cancel their booking or not.

# In[58]:


# Import libraries 

import pandas as pd
import numpy as np


# In[59]:


# Read the dataset to Python

hotel_booking = pd.read_csv('/Users/poojarangi/Downloads/hotel_bookings.csv')
hotel_booking.head(10) # Display the first 10 rows of the data


# In[60]:


# Check for Null values in columns

hotel_booking.isnull().sum() 

#Output shows 488 null values in Country, 16340 in agent and 112593 in company


# In[61]:


# Impute null values with mean because Decision Tree model wouldn't work on blank data

hotel_booking.fillna(hotel_booking.mean(), inplace=True) 


# In[62]:


# Concise summary of the dataset

hotel_booking.info() 


# In[63]:


# Descriptive Statistics 

hotel_booking.describe() 


# In[ ]:




