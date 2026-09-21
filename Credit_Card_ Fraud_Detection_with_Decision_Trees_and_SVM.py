"""
consolidate your machine learning (ML) modeling skills by using two popular classification models to identify fraudulent credit card transactions.
These models are: Decision Tree and Support Vector Machine. we will use a real dataset of credit card transactions to train each of these models.
we will then use the trained model to assess if a credit card transaction is fraudulent or not.

Objectives
After completing this lab you will be able to:

- Perform basic data preprocessing in Python
- Model a classification task using the Scikit-Learn Python APIs
- Train Suppport Vector Machine and Decision Tree models using Scikit-Learn
- Run inference and assess the quality of the trained models
"""

"""
Introduction

Imagine that you work for a financial institution and part of your job is to build a model that predicts if a credit card transaction is fraudulent or not.
You can model the problem as a binary classification problem. A transaction belongs to the positive class (1) if it is a fraud, otherwise it belongs to the negative class (0).

You have access to transactions that occured over a certain period of time. The majority of the transactions are normally legitimate and only a small fraction are non-legitimate.
Thus, typically you have access to a dataset that is highly unbalanced. This is also the case of the current dataset: only 492 transactions out of 284,807 are fraudulent
(the positive class - the frauds - accounts for 0.172% of all transactions).

This is a Kaggle dataset. You can find this "Credit Card Fraud Detection" dataset from the following link: Credit Card Fraud Detection.

To train the model, you can use part of the input dataset, while the remaining data can be utilized to assess the quality of the trained model. First, let's import the necessary 
libraries and download the dataset.
"""

!pip install pandas==2.2.3
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3


# Import the libraries we need to use in this lab
from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

import warnings
warnings.filterwarnings('ignore')

# download the dataset
url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"

# read the input data
raw_data=pd.read_csv(url)
raw_data

"""
Dataset Analysis¶
Each row in the dataset represents a credit card transaction. As shown above, each row has 31 variables.
One variable (the last variable in the table above) is called Class and represents the target variable. Your objective will be to train a model that uses the other variables
to predict the value of the Class variable. Let's first retrieve basic statistics about the target variable.
"""

# get the set of distinct classes
labels = raw_data.Class.unique()
print(labels) # [0 1]

# get the count of each class
sizes = raw_data.Class.value_counts().values
print(sizes) # [284315    492]

# plot the class value counts
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')
plt.show()
