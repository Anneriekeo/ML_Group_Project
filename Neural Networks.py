# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:09:18 2024

@author: cianr

Edited on 28-03-2024 by Annerieke
Edited on 02-04-2024 by Annerieke
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
# A trip needs at least 1 passenger to be able to receive tips, so remove all rows with less than 1 passenger
trips = trips[trips['passenger_count'] > 0]

# Tips are only recorded when a creditcard is used, so remove all rows that do not pay with creditcard
trips = trips[trips['payment_type'] == 1]

# Remove all rows containg NaN values
trips = trips.dropna()

# %% Splitting into training, validation and test dataset

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
import pandas as pd

# Dropping columns not used in training. Datetime objects dont work in training, use tip_hour instead. 
# 'store_and_fwd_flag' says whether information was at some point stored on onboard computer or not. Not important for training. 
# payment_type, MTA_tax, improvement_surcharge are all constant
# total amount includes tip so is biased; toll and airport fee are 0 fort the majority of the data; drop off is covered by pick up and distance
trips_without_dt_or_str = trips.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'store_and_fwd_flag', 'payment_type', 
                                              'mta_tax', 'improvement_surcharge', 'DOLocationID', 'extra', 'total_amount', 'tolls_amount', 'airport_fee'])

# Dataset is too large to work with, so take a random smaller sample
sample_size = 50000
trips_without_dt_or_str = trips_without_dt_or_str.sample(n = sample_size, random_state = 1)

X = trips_without_dt_or_str.drop('tip_amount', axis = 1)
Y = trips_without_dt_or_str.loc[:,'tip_amount']

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
#train_logreg = LogisticRegression(random_state=0,max_iter = 200).fit(X_train_scale,Y_train)

# Train a general Neural Network
train_nn = MLPClassifier(max_iter = 200, n_iter_no_change = 5)

train_nn.fit(X_train_scale, Y_train)

#pred_logreg = train_logreg.predict(X_val_scale)
#print("For Logistic Regression: ")
#print(classification_report(Y_val, pred_logreg))
#print ("Accuracy of logistic regression on the initial data is: ",accuracy_score(pred_logreg,Y_val))

#Predict based on the trained Neural Network using the validation data
# %%
pred_nn = train_nn.predict(X_val_scale)
print("For Neural Network: ")
print(classification_report(Y_val, pred_nn, zero_division = 0))
print ("Accuracy of NN on the initial data is: ",accuracy_score(pred_nn,Y_val))

# %%
# We want to compare the effect of different indicators on the accuracy of the neural network
# To do so, datasets without certain indicators need to be created.
# To be able to compare the accuracy, this is done after sampling the data

X_passenger = trips_without_dt_or_str.loc[:, 'passenger_count']
X_distance = trips_without_dt_or_str.loc[:, 'trip_distance']
X_pickup = trips_without_dt_or_str.loc[:, 'PULocationID']
X_fare = trips_without_dt_or_str.loc[:, 'fare_amount']
X_congestion = trips_without_dt_or_str.loc[:, 'congestion_surcharge']
X_hour = trips_without_dt_or_str.loc[:, 'tip_hour']

#splitting new datasets into train, validation and test data
(X_passenger_train,X_passenger_test,X_distance_train,X_distance_test,X_pickup_train,X_pickup_test,
 X_fare_train,X_fare_test,X_congestion_train,X_congestion_test,X_hour_train,
 X_hour_test) = train_test_split(X_passenger,X_distance,X_pickup,X_fare,X_congestion,X_hour,test_size=0.2,random_state = 1)
(X_passenger_train,X_passenger_val,X_distance_train,X_distance_val,X_pickup_train,X_pickup_val,
 X_fare_train,X_fare_val,X_congestion_train,X_congestion_val,X_hour_train,
 X_hour_val) = train_test_split(X_passenger_train,X_distance_train,X_pickup_train,X_fare_train,X_congestion_train,X_hour_train,test_size=0.25,random_state = 1)

# datapoints also need to be scaled into dataset with mean 0 and std dev = 1
X_passenger_train_scale = preprocessing.scale(X_passenger_train).reshape(-1,1)
X_passenger_val_scale = preprocessing.scale(X_passenger_val).reshape(-1,1)
X_passenger_test_scale = preprocessing.scale(X_passenger_test).reshape(-1,1)

X_distance_train_scale = preprocessing.scale(X_distance_train).reshape(-1,1)
X_distance_val_scale = preprocessing.scale(X_distance_val).reshape(-1,1)
X_distance_test_scale = preprocessing.scale(X_distance_test).reshape(-1,1)

X_pickup_train_scale = preprocessing.scale(X_pickup_train).reshape(-1,1)
X_pickup_val_scale = preprocessing.scale(X_pickup_val).reshape(-1,1)
X_pickup_test_scale = preprocessing.scale(X_pickup_test).reshape(-1,1)

X_fare_train_scale = preprocessing.scale(X_fare_train).reshape(-1,1)
X_fare_val_scale = preprocessing.scale(X_fare_val).reshape(-1,1)
X_fare_test_scale = preprocessing.scale(X_fare_test).reshape(-1,1)

X_congestion_train_scale = preprocessing.scale(X_congestion_train).reshape(-1,1)
X_congestion_val_scale = preprocessing.scale(X_congestion_val).reshape(-1,1)
X_congestion_test_scale = preprocessing.scale(X_congestion_test).reshape(-1,1)

X_hour_train_scale = preprocessing.scale(X_hour_train).reshape(-1,1)
X_hour_val_scale = preprocessing.scale(X_hour_val).reshape(-1,1)
X_hour_test_scale = preprocessing.scale(X_hour_test).reshape(-1,1)

# %%
#new neural networks

# Train Neural Networks
train_passenger_nn = MLPClassifier(max_iter = 300, tol = 0.001, n_iter_no_change = 5)
train_passenger_nn.fit(X_passenger_train_scale, Y_train)

train_distance_nn = MLPClassifier(max_iter = 300, tol = 0.001, n_iter_no_change = 5)
train_distance_nn.fit(X_distance_train_scale, Y_train)

train_pickup_nn = MLPClassifier(max_iter = 300, tol = 0.001, n_iter_no_change = 5)
train_pickup_nn.fit(X_pickup_train_scale, Y_train)

train_fare_nn = MLPClassifier(max_iter = 300, tol = 0.001, n_iter_no_change = 5)
train_fare_nn.fit(X_fare_train_scale, Y_train)

train_congestion_nn = MLPClassifier(max_iter = 300, tol = 0.001, n_iter_no_change = 5)
train_congestion_nn.fit(X_congestion_train_scale, Y_train)

train_hour_nn = MLPClassifier(max_iter = 300, tol = 0.001, n_iter_no_change = 5)
train_hour_nn.fit(X_hour_train_scale, Y_train)

#Predict based on the trained Neural Networks using the validation data
pred_passenger_nn = train_passenger_nn.predict(X_passenger_val_scale)
print("For passenger Neural Network: ")
print(classification_report(Y_val, pred_passenger_nn, zero_division = 0))
print ("Accuracy of passenger NN on the initial data is: ",accuracy_score(pred_passenger_nn,Y_val))

pred_distance_nn = train_distance_nn.predict(X_distance_val_scale)
print("For distance Neural Network: ")
print(classification_report(Y_val, pred_distance_nn, zero_division = 0))
print ("Accuracy of distance NN on the initial data is: ",accuracy_score(pred_distance_nn,Y_val))

pred_pickup_nn = train_pickup_nn.predict(X_pickup_val_scale)
print("For pickup Neural Network: ")
print(classification_report(Y_val, pred_pickup_nn, zero_division = 0))
print ("Accuracy of pickup NN on the initial data is: ",accuracy_score(pred_pickup_nn,Y_val))

pred_fare_nn = train_fare_nn.predict(X_fare_val_scale)
print("For fare Neural Network: ")
print(classification_report(Y_val, pred_fare_nn, zero_division = 0))
print ("Accuracy of fare NN on the initial data is: ",accuracy_score(pred_fare_nn,Y_val))

pred_congestion_nn = train_congestion_nn.predict(X_congestion_val_scale)
print("For congestion Neural Network: ")
print(classification_report(Y_val, pred_congestion_nn, zero_division = 0))
print ("Accuracy of congestion NN on the initial data is: ",accuracy_score(pred_congestion_nn,Y_val))

pred_hour_nn = train_hour_nn.predict(X_hour_val_scale)
print("For hour Neural Network: ")
print(classification_report(Y_val, pred_hour_nn, zero_division = 0))
print ("Accuracy of hour NN on the initial data is: ",accuracy_score(pred_hour_nn,Y_val))

# %%
X_vendor = trips_without_dt_or_str.loc[:, 'VendorID']
X_code = trips_without_dt_or_str.loc[:, 'RatecodeID']

(X_vendor_train,X_vendor_test,X_code_train,X_code_test) = train_test_split(X_vendor,X_code,test_size=0.2,random_state = 1)
(X_vendor_train,X_vendor_val,X_code_train,X_code_val) = train_test_split(X_vendor_train,X_code_train,test_size=0.25,random_state = 1)

X_vendor_train_scale = preprocessing.scale(X_vendor_train).reshape(-1,1)
X_vendor_val_scale = preprocessing.scale(X_vendor_val).reshape(-1,1)
X_vendor_test_scale = preprocessing.scale(X_vendor_test).reshape(-1,1)

X_code_train_scale = preprocessing.scale(X_code_train).reshape(-1,1)
X_code_val_scale = preprocessing.scale(X_code_val).reshape(-1,1)
X_code_test_scale = preprocessing.scale(X_code_test).reshape(-1,1)

train_vendor_nn = MLPClassifier(max_iter = 300, tol = 0.001, n_iter_no_change = 5)
train_vendor_nn.fit(X_vendor_train_scale, Y_train)

train_code_nn = MLPClassifier(max_iter = 300, tol = 0.001, n_iter_no_change = 5)
train_code_nn.fit(X_code_train_scale, Y_train)

pred_vendor_nn = train_vendor_nn.predict(X_vendor_val_scale)
print("For vendor Neural Network: ")
print(classification_report(Y_val, pred_vendor_nn, zero_division = 0))
print ("Accuracy of vendor NN on the initial data is: ",accuracy_score(pred_vendor_nn,Y_val))

pred_code_nn = train_code_nn.predict(X_code_val_scale)
print("For code Neural Network: ")
print(classification_report(Y_val, pred_code_nn, zero_division = 0))
print ("Accuracy of code NN on the initial data is: ",accuracy_score(pred_code_nn,Y_val))
    # %%
print ("Accuracy of passenger NN on the initial data is: ",accuracy_score(pred_passenger_nn,Y_val))
print ("Accuracy of distance NN on the initial data is: ",accuracy_score(pred_distance_nn,Y_val))
print ("Accuracy of pickup NN on the initial data is: ",accuracy_score(pred_pickup_nn,Y_val))
print ("Accuracy of fare NN on the initial data is: ",accuracy_score(pred_fare_nn,Y_val))
print ("Accuracy of congestion NN on the initial data is: ",accuracy_score(pred_congestion_nn,Y_val))
print ("Accuracy of hour NN on the initial data is: ",accuracy_score(pred_hour_nn,Y_val))
print ("Accuracy of vendor NN on the initial data is: ",accuracy_score(pred_vendor_nn,Y_val))
print ("Accuracy of code NN on the initial data is: ",accuracy_score(pred_code_nn,Y_val))
# %%
