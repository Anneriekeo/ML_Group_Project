# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:09:18 2024

@author: cianr
"""
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

# %% Plotting data to inspect it (tips)

# Histogram of tip amount for indication (unreadable)
plt.hist(trips['tip_amount'], bins=40, edgecolor='black')
plt.title('tips')
plt.show()

# Remove outliers to make plot readable
# Method 1: remove top 5% of entries
tips_95_percent = trips[trips['tip_amount'] < trips['tip_amount'].quantile(.95)]['tip_amount'] 
plt.hist(tips_95_percent, bins=40, edgecolor='black', color='green')
plt.title('tips without outliers (95%)')
plt.show()

# Method 2: plot all tips below or equal to 25$
tips_no_outliers = trips[trips['tip_amount'] <= 25]['tip_amount']
plt.hist(tips_no_outliers, bins=40, edgecolor='black', color='green')
plt.title('tips without outliers (<=25$)')
plt.show()

# %% Plot total amount

# Histogram of total amount for indication (unreadable)
plt.hist(trips['total_amount'], bins=40, edgecolor='black')
plt.title('Total amount')
plt.show()

# Remove outliers to make plot readable
# Method 1: remove top 1% of entries
total_95_percent = trips[trips['total_amount'] < trips['total_amount'].quantile(.99)]['total_amount'] 
plt.hist(total_95_percent, bins=40, edgecolor='black', color='green')
plt.title('Total amount without outliers (99%)')
plt.show()

# Method 2: plot all tips below or equal to 25$
total_no_outliers = trips[trips['total_amount'] <= 125]['total_amount']
plt.hist(total_no_outliers, bins=40, edgecolor='black', color='green')
plt.title('Total amount without outliers (<=125$)')
plt.show()

# %% Plot tip amount throughout the day

# Histogram of tips per hour of the day
plt.hist(trips['tip_hour'], bins=24, edgecolor='black')
plt.title('Number of tips per hour of day')
plt.show()

# Bar chart with average tip amount per hour of the day
hourly_average_tip = trips.groupby('tip_hour')['tip_amount'].mean()
hourly_average_tip.plot(kind='bar', edgecolor='black')
plt.title('Average tip amounts per hour of the day')
plt.show()
# %% Splitting into training, validation and test dataset

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report

# Dropping columns not used in training. Datetime objects dont work in training, use tip_hour instead. 
# 'store_and_fwd_flag' says whether information was at some point stored on onboard computer or not. Not important for training. 
trips_without_dt_or_str = trips.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'store_and_fwd_flag'])

X = trips_without_dt_or_str
Y = trips_without_dt_or_str['tip_amount']

# Below is used in excersize 4 because target 'class' is string. Our target is numerical so not needed. 

#label encoding is done as model accepts only numeric values
# # so strings need to be converted into labels
# LE = preprocessing.LabelEncoder()
# LE.fit(Y)
# Y = LE.transform(Y)

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