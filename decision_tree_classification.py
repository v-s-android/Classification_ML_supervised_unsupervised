'''
Explore decision tree classification, a powerful machine learning technique for making data-driven decisions. You will learn to build,
visualize, and evaluate decision trees using a real-world dataset. The dataset used in this lab is that of Drug prediction based on the health parameters of a patient.

About the dataset
Imagine that you are a medical researcher compiling data for a study. You have collected data about a set of patients, all of whom suffered from the same illness.
During their course of treatment, each patient responded to one of 5 medications, Drug A, Drug B, Drug C, Drug X and Drug Y.

Part of your job is to build a model to find out which drug might be appropriate for a future patient with the same illness. The features of this dataset are the Age,
Sex, Blood Pressure, and Cholesterol of the patients, and the target is the drug that each patient responded to.

It is a sample of a multiclass classifier, and you can use the training part of the dataset to build a decision tree, and then use it to predict the class of
an unknown patient or to prescribe a drug to a new patient.
'''
# !pip install numpy==2.2.0
# !pip install pandas==2.2.3
# !pip install scikit-learn==1.6.0
# !pip install matplotlib==3.9.3

import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

# Downloading the Data
path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
df = pd.read_csv(path)
df.head()

'''
Data Analysis and pre-processing
You should apply some basic analytics steps to understand the data better. First, let us gather some basic information about the dataset.
'''
df.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200 entries, 0 to 199
Data columns (total 6 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   Age          200 non-null    int64  
 1   Sex          200 non-null    object 
 2   BP           200 non-null    object 
 3   Cholesterol  200 non-null    object 
 4   Na_to_K      200 non-null    float64
 5   Drug         200 non-null    object 
dtypes: float64(1), int64(1), object(4)
memory usage: 9.5+ KB
'''

'''
This tells us that 4 out of the 6 features of this dataset are categorical, which will have to be converted into numerical ones to be used for modeling. 
For this, we can make use of LabelEncoder from the Scikit-Learn library.
'''
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['BP'] = label_encoder.fit_transform(df['BP'])
df['Cholesterol'] = label_encoder.fit_transform(df['Cholesterol'])
df.head()

'''
You can also check if there are any missing values in the dataset.
'''
df.isnull().sum() # prins the columns with the count of null values
'''
Age            0
Sex            0
BP             0
Cholesterol    0
Na_to_K        0
Drug           0
dtype: int64
'''

'''
To evaluate the correlation of the target variable with the input features, it will be convenient to map the different drugs to a numerical value.
Execute the following cell to achieve the same.
'''
custom_map_dict = { 'drugA':0, 'drugB':1, 'drugC':2, 'drugX':3, 'drugY':4 }
df['Drug_value'] = df['Drug'].map(custom_map_dict) # we are mapping the dictionary and creating/assigning it to a new column
df.head()

'''
You can now use the corr() function to find the correlation of the input variables with the target variable.
'''

'''
Practice question
Write the code to find the correlation of the input variables with the target variable and identify the features most significantly affecting the target.
'''
# corr() - computes the correlation matrix between all numeric columns.
df_new =  df.drop('Drug' , axis= 1) # The object column "Drug" is removed because correlation only works with numeric data.
df_new.head()
print(df_new.corr()) # Each value tells you how strongly two variables are linearly related.
print(df_new.corr()['Drug_value']) # selecting only the Drug_value wrt the other remaining features
'''
                  Age       Sex        BP  Cholesterol   Na_to_K  Drug_value
Age          1.000000  0.102027  0.054212    -0.068234 -0.063119   -0.004828
Sex          0.102027  1.000000 -0.007814    -0.008811 -0.125008   -0.098573
BP           0.054212 -0.007814  1.000000    -0.137552 -0.149312    0.372868
Cholesterol -0.068234 -0.008811 -0.137552     1.000000  0.010000    0.055629
Na_to_K     -0.063119 -0.125008 -0.149312     0.010000  1.000000    0.589120
Drug_value  -0.004828 -0.098573  0.372868     0.055629  0.589120    1.000000


Age           -0.004828
Sex           -0.098573
BP             0.372868 # this is the second best 
Cholesterol    0.055629
Na_to_K        0.589120 # this is the first best among the features
Drug_value     1.000000
Name: Drug_value, dtype: float64

This shows that the drug recommendation is mostly correlated with the Na_to_K and BP features.
'''

# alreanatively use df.drop('Drug',axis=1).corr()['Drug_value'] : this doesnot delete the Drug column rather just excludes it while running corr()

'''
We can also understand the distribution of the dataset by plotting the count of the records with each drug recommendation.
'''
drug_counts = df['Drug'].value_counts()

print(drug_counts)
'''
Drug
drugY    91
drugX    54
drugA    23
drugC    16
drugB    16
Name: count, dtype: int64
'''

plt.bar(drug_counts.index , drug_counts.values, color = 'blue')
plt.xlabel('Drug')
plt.ylabel('Count')
plt.title('Category Distribution')
plt.xticks(rotation=45)  # Rotate labels for better readability if needed
plt.show()


