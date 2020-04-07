#!/usr/bin/env python
# coding: utf-8

# In[58]:


# Import libraries 

import pandas as pd
import numpy as np


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




