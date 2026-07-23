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
'''
