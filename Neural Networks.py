# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:09:18 2024

@author: cianr

Edited on 28-03-2024 by Annerieke
"""

# %%
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

# for now only look at yellow taxi data
trips = pq.read_table('yellow_tripdata_2023-01.parquet') 
trips = trips.to_pandas() 

# trips2 = pq.read_table('fhvhv_tripdata_2023-01.parquet') 
# trips2 = trips2.to_pandas() 

# %%  Cleaning up dataset and adding tip hour column

# Add tip hour column
trips['tip_hour'] = trips['tpep_dropoff_datetime'].dt.hour

# Make column name vector
columns = trips.columns

# Check if there are entries with negative amounts for columns with numerical values
for col in columns.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'store_and_fwd_flag']):
    print(len(trips[trips[col] < 0]))

#length of dataframe before removing negative entries
len_before = len(trips)

# There are rows with negative values. Removing rows with negative entries for key columns
for col in columns.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'store_and_fwd_flag']):
    trips = trips.drop(trips[trips[col] < 0].index)

# Check if removal succeeded
print('---------------------')
for col in columns.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'store_and_fwd_flag']):
    print(len(trips[trips[col] < 0]))

# No more negative entries. 

print("Number of rows removed: ", len_before - len(trips))

#%%
# Remove all rows containg NaN values
trips = trips.dropna()

# %% Splitting into training, validation and test dataset

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
import pandas as pd

# Dropping columns not used in training. Datetime objects dont work in training, use tip_hour instead. 
# 'store_and_fwd_flag' says whether information was at some point stored on onboard computer or not. Not important for training. 
trips_without_dt_or_str = trips.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'store_and_fwd_flag'])

# Dataset is too large to work with, so take a random smaller sample
sample_size = 50000
trips_without_dt_or_str = trips_without_dt_or_str.sample(n = sample_size)

X = trips_without_dt_or_str.drop('tip_amount', axis = 1)
Y = trips_without_dt_or_str['tip_amount']

# Below is used in excersize 4 because target 'class' is string. Our target is numerical so not needed. 

#label encoding is done as model accepts only numeric values
# # so strings need to be converted into labels
LE = preprocessing.LabelEncoder()
LE.fit(Y)
Y = LE.transform(Y)

#splitting dataset into train, validation and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state = 1)
X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size=0.25,random_state = 1)

# datapoints also need to be scaled into dataset with mean 0 and std dev = 1
X_train_scale = preprocessing.scale(X_train)
X_val_scale = preprocessing.scale(X_val)
X_test_scale = preprocessing.scale(X_test)

#Output the number of data points in training, validation, and test dataset.
print("Datapoints in Training set:",len(X_train))
print("Datapoints in validation set:",len(X_val))
print("Datapoints in Test set:",len(X_test))
# %%
# Training NN models and Linear regression models for comparison
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#First a general model
train_logreg = LogisticRegression(random_state=0,max_iter = 100).fit(X_train_scale,Y_train)

# Train a general Neural Network
train_nn = MLPClassifier(max_iter = 100)

train_nn.fit(X_train_scale, Y_train)

pred_logreg = train_logreg.predict(X_val_scale)
print("For Logistic Regression: ")
print(classification_report(Y_val, pred_logreg))
print ("Accuracy of logistic regression on the initial data is: ",accuracy_score(pred_logreg,Y_val))

#Predict based on the trained Neural Network using the validation data

pred_nn = train_nn.predict(X_val_scale)
print("For Neural Network: ")
print(classification_report(Y_val, pred_nn))
print ("Accuracy of NN on the initial data is: ",accuracy_score(pred_nn,Y_val))
# %%
