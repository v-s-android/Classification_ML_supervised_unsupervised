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

"""
As shown above, the Class variable has two values: 0 (the credit card transaction is legitimate) and 1 (the credit card transaction is fraudulent).
Thus, you need to model a binary classification problem. Moreover, the dataset is highly unbalanced, the target variable classes are not represented equally. 
This case requires special attention when training or when evaluating the quality of a model. One way of handing this case at train time is to bias the model 
to pay more attention to the samples in the minority class. The models under the current study will be configured to take into account the class weights of the samples at train/fit time.

It is also prudent to understand which features affect the model in what way. We can visualize the effect of the different features on the model using the code below.

"""
correlation_values = raw_data.corr()['Class'].drop('Class')
correlation_values.plot(kind='barh', figsize=(10, 6))

"""
Dataset Preprocessing

You will now prepare the data for training. You will apply standard scaling to the input features and normalize them using "L1" norm for the training models to converge quickly.
As seen in the data snapshot, there is a parameter called Time which we will not be considering for modeling. Hence, features 2 to 30 will be used as input features and feature 31,
i.e. Class will be used as the target variable.
"""

# standardize features by removing the mean and scaling to unit variance
raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(raw_data.iloc[:, 1:30]) # [:, 1:30] means all rows and 1 to 30 columns
data_matrix = raw_data.values

# X: feature matrix (for this analysis, we exclude the Time variable from the dataset)
X = data_matrix[:, 1:30]
print("feature matrix: ",X)

# y: labels vector
y = data_matrix[:, 30] # the last column
print("labels vector: ", y)

# data normalization
X = normalize(X, norm="l1")
print("normalize: ", X)

"""
Dataset Train/Test Split¶
Now that the dataset is ready for building the classification models, you need to first divide the pre-processed dataset into a subset to be used for training
the model (the train set) and a subset to be used for evaluating the quality of the model (the test set).
"""

X_train, X_test, y_train, y_test = train_test_split( X , y , test_size = 0.3, random_state = 42)

"""
Build a Decision Tree Classifier model with Scikit-Learn
Compute the sample weights to be used as input to the train routine so that it takes into account the class imbalance present in this dataset.
"""
w_train = compute_sample_weight('balanced', y_train)

"""
Using these sample weights, we may train the Decision Tree classifier.
"""

# for reproducible output across multiple function calls, set random_state to a given integer value
dt = DecisionTreeClassifier(max_depth=4, random_state=35)
dt.fit(X_train, y_train, sample_weight = w_train)

"""
Build a Support Vector Machine model with Scikit-Learn
Unlike Decision Trees, we do not need to initiate a separate sample_weight for SVMs. We can simply pass a parameter in the scikit-learn function.
"""

# for reproducible output across multiple function calls, set random_state to a given integer value
svm = LinearSVC(class_weight='balanced', random_state=31, loss = "hinge", fit_intercept = False)
svm.fit(X_train, y_train)

"""
Evaluate the Decision Tree Classifier Models
the below computes the probabilities of the test samples belonging to the class of fraudulent transactions.
"""

y_pred_dt = dt.predict_proba(X_test)[:, 1]

"""
Using these probabilities, we can evaluate the Area Under the Receiver Operating Characteristic Curve (ROC-AUC) score as a metric of model performance.
The AUC-ROC score evaluates your model's ability to distinguish positive and negative classes considering all possible probability thresholds. 
The higher its value, the better the model is considered for separating the two classes of values.
"""
roc_auc_dt = roc_auc_score(y_test , y_pred_dt)
print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dt)) # Decision Tree ROC-AUC score : 0.939

"""
Evaluate the Support Vector Machine Models
the below computes the probabilities of the test samples belonging to the class of fraudulent transactions.
"""
y_pred_svm = svm.decesion_function(X_test)

# evaluate the accuracy of SVM on the test set in terms of the ROC-AUC score.
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm)) #SVM ROC-AUC score: 0.986

# practice
# Use the corr() function to find the top 6 features of the dataset to train the models on.
correlation_values = abs(raw_data.corr()['Class']).drop('Class')
correlation_values = correlation_values.sort_values(ascending=False)[:6]
correlation_values
"""
V17    0.326481
V14    0.302544
V12    0.260593
V10    0.216883
V16    0.196539
V3     0.192961
"""
# Using only these 6 features, modify the input variable for training.
X = data_matrix[:,[3,10,12,14,16,17]]


