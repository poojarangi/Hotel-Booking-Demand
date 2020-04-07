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


# In[70]:


# Finding out significant variables for modeling

cancelation_corr = hotel_booking.corr()["is_canceled"]
cancelation_corr.abs().sort_values(ascending=False)[1:] 


# In[71]:


# factorize the independent categorical variables for Decision Tree

hotel_booking['arrival_date_month'],_ = pd.factorize(hotel_booking['arrival_date_month'])
hotel_booking.head() 


# In[72]:


# Split the data into independent(features) and dependent (target) variables 

feature_cols = ['lead_time','total_of_special_requests','required_car_parking_spaces','booking_changes','previous_cancellations',
        'is_repeated_guest','agent','adults','previous_bookings_not_canceled','days_in_waiting_list','adr',
        'babies','stays_in_week_nights','company','arrival_date_year','arrival_date_week_number','arrival_date_day_of_month',
        'children','stays_in_weekend_nights'] #features

x = hotel_booking[feature_cols] # features
y = hotel_booking[['is_canceled']]  # target variable

print(x.head()) 
print("\n")  
print(y.head()) 

x.info()
print("\n")
y.info() 


# In[73]:


# Split the data into train and test data
#testing data size is of 25% of entire data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=0)

print("Rows and columns of training data",x_train.shape ) 

print("Rows and columns of test data", x_test.shape) 


# In[74]:


# create decision tree classifier object
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 
clf = DecisionTreeClassifier()


# In[75]:


# Train Decision Tree Classifer

clf = clf.fit(x_train,y_train)


# In[76]:


#Predict the response for test dataset

y_pred = clf.predict(x_test)


# In[77]:


# Model Accuracy, how often is the classifier correct?

from sklearn.metrics import accuracy_score 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[78]:


# Print the confusion matrix 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred)) 


# In[79]:


# Calculate error values in the model 

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[80]:


# Get predicted probabilities

y_score1 = clf.predict_proba(x_test)[:,1]


# In[81]:


# Calculate the ROC value 

from sklearn.metrics import roc_curve, roc_auc_score
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_score1)
print('roc_auc_score for DecisionTree: ', roc_auc_score(y_test, y_score1)) 


# In[82]:


# Plot ROC curve

plt.title('Receiver Operating Characteristic - DecisionTree')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7") 
plt.ylabel('Sensitivity')
plt.xlabel('1-Specificity')
plt.show() 


# In[ ]:




