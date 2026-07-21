'''
 different strategies of Multi-class classification and implement the same on a real-world dataset.

- Understand the use of one-hot encoding for categorical variables.
- Implement logistic regression for multi-class classification using One-vs-All (OvA) and One-vs-One (OvO) strategies.
- Evaluate model performance using appropriate metrics.

 '''

# Import Necessary Libraries
!pip install numpy==2.2.0
!pip install pandas==2.2.3
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3
!pip install seaborn==0.13.2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
df = pd.read_csv(url)
df.head()

# Exploratory Data Analysis
'''
Visualize the distribution of the target variable to understand the class balance.
'''
sns.countplot(y = "NObeyesdad" , data = df)
plt.title("Distribution of Obesity Levels")
plt.show()

# Exercise 1¶
# Check for null values, and display a summary of the dataset (use .info() and .describe() methods).
print(df.isnull().sum()) # displays the count of null values in each column
'''
Gender                            0
Age                               0
Height                            0
Weight                            0
family_history_with_overweight    0
FAVC                              0
FCVC                              0
NCP                               0
CAEC                              0
SMOKE                             0
CH2O                              0
SCC                               0
FAF                               0
TUE                               0
CALC                              0
MTRANS                            0
NObeyesdad                        0
dtype: int64
'''
print(df.describe()) # Descriptive statistics for numerical columns.
print(df.info()) # Dataset info including column names, data types, and memory usage.

# Preprocessing the data
'''
Feature scaling
Scale the numerical features to standardize their ranges for better model performance.

Standardization of data is important to better define the decision boundaries between classes by making sure that the feature variations are in similar scales.
The data is now ready to be used for training and testing.
'''
# creating a list of columns that have float datatype
float_columns = df.select_dtypes(include = ['float64']).columns.tolist() 

std_scaler = StandardScaler()
scaled_columns = std_scaler.fit_transform(df[float_columns]) #  standardizing the columns 

# Converting to a DataFrame
sacled_df = pd.DataFrame(scaled_columns , columns = std_scaler.get_feature_names_out(float_columns)) # using the same names as it is in df

new_df = pd.concat([df.drop(columns = float_columns), sacled_df ],axis = 1) # dropping and concatinating the old df columns with the new standardized columns

'''
One-hot encoding 
Convert categorical variables into numerical format using one-hot encoding.
'''

# get the list of categorical variables
categorical_columns = new_df.select_dtypes(include = ['object']).columns.tolist()
categorical_columns.remove("NObeyesdad") # Remove the target column name from the list

print(categorical_columns) # ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(df[categorical_columns])
'''
Color
Red
Blue      
Green
Red
'''
# becomes
'''
Color_Blue	Color_Green	Color_Red
0	        0	              1
1	        0	              0
0        	1	              0
0	        0	              1
'''
# then we drop ='first'
'''
the encoder drops the first category.
Green	Red
0	  1
0	  0
1	  0

Now:

(0,0) means Blue
(0,1) means Red
(1,0) means Green

You still have enough information to distinguish all categories while avoiding redundant columns.
'''

# Creating a dataframe
encoded_df = pd.DataFrame(encoded_features, columns =  encoder.get_feature_names_out(categorical_columns)) # makinf use of the same column name
prepped_df = pd.concat( [new_df.drop(columns = categorical_columns), encoded_df] , axis = 1) # drop the categorical_columns and concatinate encoded dataframe at the end

prepped_df.head()

'''
Encode the target variable
'''
# Encoding the target variable
prepped_df['NObeyesdad'] = prepped_df['NObeyesdad'].astype('category').cat.codes # converting into category codes
prepped_df.head()

'''
Separate the input and target data
'''
y = prepped_df['NObeyesdad'] # assign the target feature 

X = prepped_df.drop( 'NObeyesdad' , axis = 1) # then delete it from dataframe

'''
Model training and evaluation
Splitting the data set
Split the data into training and testing subsets.
'''

X_train, X_test, y_train, y_test = train_test_split( X , y , test_size = 0.20, random_state = 1 , stratify = y )
# stratify=y = This makes sure the class proportions in your training and testing sets stay similar to those in the original dataset.

'''
Logistic Regression with One-vs-All
Train a logistic regression model using the One-vs-All strategy and evaluate its performance.
'''

model_ova = LogisticRegression(multi_class = "ovr" , max_iter = 1000)
model_ova.fit(X_train, y_train)

# You can now evaluate the accuracy of the trained model as a measure of its performance on unseen testing data.
y_pred_ova = model_ova.predict(X_test)

print("One-vs-All (OvA) Strategy")
print(f" the accuracy is {np.round ( 100 * accuracy_score(y_test , y_pred_ova), 2)} %")
'''
One-vs-All (OvA) Strategy
 the accuracy is 76.12 %
'''

'''
Logistic Regression with OvO
Train a logistic regression model using the One-vs-One (OvO) strategy and evaluate its performance.
'''
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter = 1000))
model_ovo.fit(X_train, y_train)

y_ovo_pred = model_ovo.predict(X_test)
print(f" the accuracy is {np.round( 100 * accuracy_score(y_test, y_ovo_pred),2)} % ")
'''
One-vs-One (OvO) Strategy
 the accuracy is 92.2 % 
'''

'''
Exercises
Q1. Experiment with different test sizes in the train_test_split method (e.g., 0.1, 0.3) and observe the impact on model performance.
'''
for test_size in [0.1, 0.3]:
    X_train, X_test, y_train , y_test = train_test_split( X , y , test_size= test_size, random_state= 42, stratify= y)
    model_ova.fit(X_train, y_train)
    y_ova_pred = model_ova.predict(X_test)
    print(f"the accuracy of test_size: {test_size} is {np.round(100 * accuracy_score(y_test, y_ova_pred),2)}%")
'''
the accuracy of test_size: 0.1 is 75.94%
the accuracy of test_size: 0.3 is 74.92% 
'''

'''
Q2. Plot a bar chart of feature importance using the coefficients from the One vs All logistic regression model. Also try for the One vs One model.
'''
# Feature importance
feature_importance = np.mean(np.abs(model_ova.coef_), axis=0)
plt.barh(X.columns, feature_importance)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()

# For One vs One model
# Collect all coefficients from each underlying binary classifier
coefs = np.array([est.coef_[0] for est in model_ovo.estimators_]) # there are lot of arrays , we take the first value and create an array of first values 
'''
[[ 5.26958812e-01 -2.10346113e+00  7.99657915e+00 -3.23125743e-01
  -1.01076741e-01 -2.12634946e-01  8.65326589e-02 -1.35733425e-01
   5.48032451e-01  1.50991813e-01  2.85700123e-01 -1.19032100e+00
  -7.35719837e-01 -3.39310154e-01  7.67372879e-01  2.25980478e-01
   4.28861917e-01 -2.34849885e-01 -1.94682995e-01  7.47849030e-01
   2.28597955e-01 -1.29592814e-01  5.38003813e-01] # end of first array
 [ 7.45066448e-01 -7.91417826e-01  4.53935137e+00 -2.47267218e-01
  -5.00771023e-01 -1.72456665e-01 -6.79434986e-02 -1.92749959e-01
  -2.69039086e-01  9.67800833e-01  2.29856446e-01 -7.37531871e-01
   7.30493853e-01 -5.04824563e-02  1.36637777e-01 -1.12790975e-01
   1.09688769e-01 -2.33972601e-01  1.28424481e-01  0.00000000e+00
   1.60166091e-02  1.91889385e-01  9.57695805e-02] # end of second array
...
'''

# Now take the mean across all those classifiers
feature_importance = np.mean(np.abs(coefs), axis=0) # abs converts the values into positive floats and then the mean is calculated

# Plot feature importance
plt.barh(X.columns, feature_importance)
plt.title("Feature Importance (One-vs-One)")
plt.xlabel("Importance")
plt.show()

'''
Q3. Write a function obesity_risk_pipeline to automate the entire pipeline:

Loading and preprocessing the data
Training the model
Evaluating the model
The function should accept the file path and test set size as the input arguments.
'''

def obesity_risk_pipeline(data_path, test_size=0.2):
    # Load data
    df = pd.read_csv(data_path)

    # Standardizing continuous numerical features
    numerical_columns = df.select_dtypes(include = ["float64"]).columns.tolist()
    std_scaler = StandardScaler()
    scalered_df = std_scaler.fit_transform(df[numerical_columns])

    # Converting to a DataFrame
    std_scaler_df = pd.DataFrame(scalered_df , columns = std_scaler.get_feature_names_out(numerical_columns))

    # Combining with the original dataset
    new_df = pd.concat( [df.drop(columns=numerical_columns),std_scaler_df ], axis = 1 )

    # Identifying categorical columns
    # covert catogorical columns into numercial column values
    categorical_columns = new_df.select_dtypes(include = ['object']).columns.tolist()
    categorical_columns.remove('NObeyesdad') # Exclude target column

     # Applying one-hot encoding
    encoder = OneHotEncoder(sparse_output= False, drop='first') # drops the first category : ex- Blue, red, Green 
    encoded_df = encoder.fit_transform(df[categorical_columns])

    # Converting to a DataFrame
    encoded_df_new = pd.DataFrame(encoded_df , columns = encoder.get_feature_names_out(categorical_columns))
    prepped_df = pd.concat( [new_df.drop(columns = categorical_columns) , encoded_df_new ], axis = 1)

    # Encoding the target variable
    prepped_df['NObeyesdad'] = prepped_df['NObeyesdad'].astype('category').cat.codes

    # Preparing final dataset
    y = prepped_df['NObeyesdad']
    X = prepped_df.drop('NObeyesdad', axis = 1)

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size= test_size, random_state=42, stratify= y )

    model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
    model_ova.fit(X_train, y_train)

    y_ova_pred = model_ova.predict(X_test)
    print(f"the model_ova accuracy is {np.round(100 * accuracy_score(y_test, y_ova_pred))}%")

    model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
    model_ovo.fit(X_train, y_train)

    y_ovo_pred = model_ovo.predict(X_test)
    print(f"the model_ovo accuracy is {np.round(100 * accuracy_score(y_test, y_ovo_pred))}%")

    # Training and evaluation
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


obesity_risk_pipeline("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv", test_size=0.2)

'''
the model_ova accuracy is 76.0%
the model_ovo accuracy is 92.0%
Accuracy: 0.8794326241134752
'''

