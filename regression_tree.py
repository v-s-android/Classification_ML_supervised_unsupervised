'''
The dataset includes information about taxi tip and was collected and provided to the NYC Taxi and Limousine Commission (TLC) by technology providers authorized under
the Taxicab & Livery Passenger Enhancement Programs (TPEP/LPEP). You will use the trained model to predict the amount of tip paid.

Objectives¶

- Perform basic data preprocessing using Scikit-Learn
- Model a regression task using Scikit-Learn
- Train a Decision Tree Regressor model
- Run inference and assess the quality of the trained models

The dataset used in this exercise session is a subset of the publicly available TLC Dataset (all rights reserved by Taxi & Limousine Commission (TLC), City of New York).
The prediction of the tip amount can be modeled as a regression problem.
To train the model you can use part of the input dataset and the remaining data can be used to assess the quality of the trained model.
'''

# !pip install numpy
# !pip install pandas
# !pip install matplotlib
# !pip install scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
df = pd.read_csv(url)
df.head()

'''
Each row in the dataset represents a taxi trip. As shown above, each row has 13 variables. One of the variables is tip_amount which will be the target variable.
Your objective will be to train a model that uses the other variables to predict the value of the tip_amount variable.

To understand the dataset a little better, let us plot the correlation of the target variable against the input variables.
'''

correlation_values = df.corr()['tip_amount'].drop('tip_amount') # this will give correlation of all inout variables wrt target variable
print(correlation_values)
correlation_values.plot( kind = 'barh')
'''
VendorID                      NaN
passenger_count          0.015081
trip_distance            0.101819
RatecodeID               0.094075
store_and_fwd_flag       0.000320
PULocationID            -0.023086
DOLocationID             0.024348
payment_type                  NaN
fare_amount              0.200638
mta_tax                 -0.054488
tolls_amount             0.116172
improvement_surcharge   -0.000727
Name: tip_amount, dtype: float64
'''

'''
Dataset Preprocessing
You will now prepare the data for training by applying normalization to the input features.
'''

# extract and Convert the tip_amount dtype to float32
y = df['tip_amount'].values.astype('float32') # we are getting the values of tip_amount
print("tip_amount-y : ", y) # 1 D array tip_amount-y :  [16.54 16.19 12.   ...  8.   16.19  4.13]

X_df = df.drop('tip_amount', axis = 1)
X = X_df.values # getting the values
print("X values :  ", X)
'''
2D array X values :   [[ 2.    1.   17.63 ...  0.5   6.94  1.  ]
 [ 2.    1.   19.52 ...  0.5   6.94  1.  ]
 [ 2.    1.   17.81 ...  0.5   6.94  1.  ]
 ...
 [ 2.    1.   17.31 ...  0.5   6.94  1.  ]
 [ 2.    1.   17.28 ...  0.5   6.94  1.  ]
 [ 2.    1.   16.82 ...  0.5   6.94  1.  ]]
 '''
X = normalize( X , axis=1, norm = 'l1', copy = False ) # For L1 normalization, the sum of the absolute values in each row becomes 1.
print("normalized X :", X)
'''
normalize() is commonly used in data preprocessing (from sklearn.preprocessing) to normalize each row (sample) of the dataset so that the sum of its absolute values equals 1.

normalized X : [[0.00501165 0.00250583 0.04417771 ... 0.00125291 0.01739043 0.00250583]
 [0.00422869 0.00211434 0.04127199 ... 0.00105717 0.01467355 0.00211434]
 [0.0070609  0.00353045 0.06287732 ... 0.00176523 0.02450132 0.00353045]
 ...
 [0.00537996 0.00268998 0.04656355 ... 0.00134499 0.01866846 0.00268998]
 [0.00427606 0.00213803 0.03694518 ... 0.00106902 0.01483794 0.00213803]
 [0.00494731 0.00247366 0.04160689 ... 0.00123683 0.01716717 0.00247366]]
'''

'''
Dataset Train/Test Split
Now that the dataset is ready for building the classification models, you need to first divide the pre-processed dataset into a subset to be used for training the 
model (the train set) and a subset to be used for evaluating the quality of the model (the test set).
'''

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3 , random_state = 1)

'''
Build a Decision Tree Regressor model with Scikit-Learn
Regression Trees are implemented using DecisionTreeRegressor.

The important parameters of the model are:

criterion: The function used to measure error, we use 'squared_error'.

max_depth - The maximum depth the tree is allowed to take; we use 8.
'''
from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor(criterion = 'mean_squared_error', max_depth = 4, random_state = 1)

dt_reg.fit(X_train , y_train)

'''
To evaluate our dataset we will use the score method of the DecisionTreeRegressor object providing our testing data, this number is the R^2 value which indicates
the coefficient of determination. We will also evaluate the Mean Squared Error of the regression output with respect to the test set target values. High R^2 score
and low  MSE values are expected from a good regression model.
'''

y_dt_pred = dt_reg.predict(X_test)

mse_score = mean_squared_error(y_test, y_dt_pred)
print('MSE score : {0:.3f}'.format(mse_score))

r2_score = dt_reg.score(X_test,y_test)
print('R^2 score : {0:.3f}'.format(r2_score))

print(f" TEST: the mean squared error(MSE): {np.round(mse_score, 3)}") # same answer as below
print(f" TEST: R^2 score: {np.round(r2_score, 3)}") # same answer as below

'''
MSE score : 24.555
R^2 score : 0.028
'''
