"""
K-Nearest Neighbors Classifier:

Use K-Nearest neighbors to classify data
Apply KNN classifier on a real world data set
"""

#install
!pip install numpy==2.2.0
!pip install pandas==2.2.3
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3
!pip install seaborn==0.13.2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
"""
About the data set
Imagine a telecommunications provider has segmented its customer base by service usage patterns, categorizing the customers into four groups. If demographic data can be used to predict group membership, the company can customize offers for individual prospective customers. It is a classification problem. That is, given the dataset, with predefined labels, we need to build a model to be used to predict class of a new or unknown case.

The example focuses on using demographic data, such as region, age, and marital, to predict usage patterns.

The target field, called "custcat", has four possible service categories that correspond to the four customer groups, as follows:

Basic Service
E-Service
Plus Service
Total Service
Our objective is to build a classifier to predict the service category for unknown cases. We will use a specific type of classification called K-nearest neighbors.
"""

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
df.head()

# Data Visualization and Analysis
# Let us first look at the class-wise distribution of the data set.
df['custcat'].value_counts()
# custcat
# 3    281
# 1    266
# 4    236
# 2    217
# Name: count, dtype: int64
"""
Hence, we can say that we have records of 281 customers who opt for Plus Services, 266 for Basic-services, 236 for Total Services, and 217 for E-Services. 
It can thus be seen that the data set is mostly balanced between the different classes and requires no special means of accounting for class bias.

We can also visualize the correlation map of the data set to determine how the different features are related to each other.
"""
correlation_matrix = df.corr()

plt.figure(figsize = (10, 8))
sns.heatmap(correlation_matrix , annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

"""
As is visible from the correlation map, some features have beeter correlation among them than others, basically indicating the depth of relationship between the two features.
What is of interest to us is the correlation of the target feature, i.e. custcat with all the other features. 
This will help us identify which features should be focussed on for modeling and which ones can be ignored.

The following code snippet will give us a list of features sorted in the descending order of their absolute correlation values with respect to the target field.
"""
correlation_values = abs(df.corr()['custcat'].drop('custcat')).sort_values(ascending=False)
print(correlation_values)
# ed         0.193864
# tenure     0.166691
# income     0.134525
# employ     0.110011
# marital    0.083836
# reside     0.082022
# address    0.067913
# age        0.056909
# region     0.023771
# retire     0.008908
# gender     0.004966
# Name: custcat, dtype: float64
# This shows us that the features retire and gender have the least effect on custcat while ed and tenure have the most effect.

"""
Separate the input and target features
Now, we can separate the data into the input data set and the target data set.
"""
X = df.drop('custcat',axis = 1) # dropping the custcat and assigning the resultant table to X
y = df['custcat'] # assinging custcat column to y

"""
Normalize Data
Data normalization is important for the KNN model.

KNN makes predictions based on the distance between data points (samples), i.e. for a given test point, the algorithm finds the k-nearest neighbors by measuring the distance between 
the test point and other data points in the dataset. By normalizing / standardizing the data, you ensure that all features contribute equally to the distance calculation.
Since normalization scales each feature to have zero mean and unit variance, it puts all features on the same scale (with no feature dominating due to its larger range).

This helps KNN make better decisions based on the actual relationships between features, not just on the magnitude of their values.
"""

X_norm = StandardScaler().fit_transform(X)

"""
Train Test Split
Now, you should separate the training and the testing data. You can retain 20% of the data for testing purposes and use the rest for training. 
Assigning a random state ensures reproducibility of the results across multiple executions.
"""
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

"""
KNN Classification¶
Once the data is in place, we can now execute the training of the model.

-> Training
Initially, you may start by using a small value as the value of k, say k = 4.
"""
k = 4
#Train Model and Predict  
knn_classifier = KNeighborsClassifier(n_neighbors = k)
knn_model = knn_classifier.fit(X_train,y_train)

"""
Predicting
Once the model is trained, we can now use this model to generate predictions for the test set.
"""
y_hat = knn_model.predict(X_test)

"""
Accuracy evaluation
In multilabel classification, accuracy classification score is a function that computes subset accuracy. 
This function is equal to the jaccard_score function. Essentially, it calculates how closely the actual labels and predicted labels are matched in the test set.
"""
print("Test set Accuracy: ", accuracy_score(y_test, y_hat)) # Test set Accuracy:  0.32



