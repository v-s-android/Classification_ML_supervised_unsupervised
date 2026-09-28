"""
- Use Principal Component Analysis (PCA) to project 2-D data onto its principal axes
- Use PCA for feature space dimensionality reduction
- Relate explained variance to feature importance and noise reduction
"""

"""
Introduction
-  how to implement two important applications of PCA

1. The first application illustrates how you can use PCA to project 2-D data onto its principal axes, meaning the two orthogonal directions that explain most of the variance in your data.

2. For the second application, you will use PCA to project higher dimensional data down to a lower dimensional feature space. This is an example of dimension reduction,a powerful technique that has
multiple benefits, including reducing your model-building computational load and, in many cases, the accuracy of your model. PCA can help you filter out redundant, linearly correlated variables 
and reduce the amount of noise in your data.
"""

"""
Part I: Using PCA to project 2-D data onto its principal axes
Here, you will illustrate how you can use PCA to transform your 2-D data to represent it in terms of its principal axes - the projection of your data onto the two orthogonal directions that explain most of the variance in your data. Let's see what all of this means.
"""

!pip install numpy==2.2.0
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

"""
Create dataset
Next you'll create a 2-dimensional dataset containing two linearly correlated features.

You'll use a bivariate normal distribution.

Both features, X1 and X2, will have zero mean and a covariance given by the (symmetric) covariance matrix:
{3 2}
{2 2}
Here, the diagonal elements define the variances of X1 and X2 (3 and 2, respectively), while the off-diagonal element is the covariance (2) between X1 and X2, 
which expresses how similarly these features vary.
"""

np.random.seed(42)
mean = [0 , 0]
covariance_matrix = [[3,2] , [2,2]]
X = np.random.multivariate_normal(mean = mean, cov = covariance_matrix, size = 200)

# Exercise 1. Visualize the relationship between the two features.

plt.figure()
plt.scatter(X[:, 0], X[:, 1], edgecolor = 'k', alpha = 0.7)
plt.title("Scatter Plot of Bivariate Normal Distribution")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axis('equal')
plt.grid(True)
plt.show()

"""
Perform PCA on the dataset
Next, you'll initialize a 2-component PCA model with default parameters and then fit and transform the feature space in one step.
"""
 
