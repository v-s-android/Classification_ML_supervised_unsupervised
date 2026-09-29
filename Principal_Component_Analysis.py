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
pca_model = PCA(n_components = 2)
X_pca = pca_model.fit_transform(X)

"""
Get the principal components from the model.
The principal components are the principal axes, represented in feature space coordinates, which align with the directions of maximum variance in your data.
"""
components = pca_model.components_
print(components)
"""
array([[ 0.78215821,  0.62307987],
       [-0.62307987,  0.78215821]])
"""
# The principal components are sorted in decreasing order by their explained variance, which can be expressed as a ratio:
pca_model.explained_variance_ratio_ 
"""
array([0.9111946, 0.0888054])
"""

# Exercise 2. What percentage of the variance in the data is explained by the first principal component?
"""
You can see that the first component explains over 90% of the variance in the data, while the second explains about 9%.
"""

"""
Display the results
Here, you'll use a scatterplot to display the data points in their original feature space, X1, X2.

You'll also plot the projections of the data points onto their principal component directions.

It's a bit technical, requiring some understanding of linear algebra, but the outcome will be instructive.

Let's see how this works.

Project the data onto its principal component axes
The projection of the data onto a given principal component yields the coordinates of each of the data points along that component's direction.

The new coordinates are given by the dot products of each point's coordinates with the given PCA component.

Specifically, the projections are given by:
"""
projection_pc1 = np.dot(X , components[0])
print(projection_pc1)
"""
[-1.0617152  -1.37340572  0.49856211 -3.36772678  1.00615406  0.98669923
 -0.52912477  3.68023218  2.16511245  1.9301144  -3.13161279 -0.15343099
  1.1633431   2.46058023  1.28088434  1.29702817  0.02198647 -1.76458463...
"""
projection_pc2 = np.dot(X , components[1])
print(projection_pc2)
"""
[-0.08118816  1.02193982 -0.15991079  0.54107197  0.34944934 -0.31803507
 -1.26177825 -0.4082784   0.18694605 -0.95405526 -0.11892331 -0.94194855
  0.06209042  0.22475232 -0.20566396  1.21388449 -0.70061505 -0.79119198...
"""
# Now that you have these coordinates, you can use them to represent the projections of each data point along the principal directions in the original feature space.

"""
As we know this is the components:
array([[ 0.78215821,  0.62307987],
       [-0.62307987,  0.78215821]])
"""

x_pc1 = projection_pc1 * components[0][0]
"""
x_pc1:  [-0.83042927 -1.07422057  0.38995445 -2.63409516  0.78697166  0.77175491
 -0.41385929  2.87852383  1.69346049  1.50965483 -2.44941667 -0.12000731
  0.90991836  1.92456304  1.00185421  1.01448124  0.0171969  -1.38018436
"""
y_pc1 = projection_pc1 * components[0][1]
"""
y_pc1:  [-0.66153337 -0.85574146  0.31064402 -2.09836277  0.62691434  0.61479243
 -0.32968699  2.2930786   1.34903799  1.20261543 -1.9512449  -0.09559976
  0.72485567  1.53313802  0.79809325  0.80815215  0.01369933 -1.09947716
"""
x_pc2 = projection_pc2 * components[1][0]
"""
x_pc2:  [ 0.05058671 -0.63675013  0.0996372  -0.33713106 -0.21773485  0.19816125
  0.78618863  0.25439006 -0.11648232  0.59445263  0.07409872  0.58690918
 -0.03868729 -0.14003865  0.12814507 -0.75634699  0.43653913  0.4929758
"""
y_pc2 = projection_pc2 * components[1][1]
"""
y_pc2:  [-0.06350199  0.79931862 -0.12507554  0.42320389  0.27332467 -0.24875374
 -0.98691022 -0.31933831  0.14622139 -0.74622216 -0.09301684 -0.73675279
  0.04856454  0.17579188 -0.16086175  0.94944972 -0.54799181 -0.61883731
"""
# Plot the results

# Plot original data
plt.figure()
plt.scatter(X[:,0] , X[:,1], label = 'Original Data', ec='k' , s = 50, alpha=0.7)

# Plot the projections along PC1 and PC2
plt.scatter(x_pc1, y_pc1, c ='r', ec='k', marker='X', alpha = 0.5, s = 70, label = 'Projection onto PC 1' )
plt.scatter(x_pc2, y_pc2, c='b', ec='k', marker='X', alpha = 0.5, s = 70, label = 'Projection onto PC 2')
plt.title()
plt.xlabel()
plt.ylabel()
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()

"""
now you can see what the the principal coordinates mean.

The data varies in two main directions.

The first direction, in red, is aligned in the direction having the widest variation.
"""

"""
Exercise 3. Describe the second direction.

The second direction, in blue, is perpendicular to first and has a lower variance.
"""

"""
Part II. PCA for feature space dimensionality reduction
For this second application, you'll use PCA to project the four-dimensional Iris feature data set down onto a two-dimensional feature space.

This will have the added benefit of enabling you to visualize some of the most important structures in the dataset.

Load and preprocess Iris data
Let's start by loading the iris data and standardizing is features.
"""
# Load the Iris dataset
iris = datasets.load_iris() # see "from sklearn import datasets"
X = iris.data
y = iris.target

target_names = iris.target_names
print(target_names) # array(['setosa', 'versicolor', 'virginica'], dtype='<U10')

# Standardize the data
scalar = StandardScalar()
X_scaled = scalar.fit_transform(X)

# Exercise 5. Initialize a PCA model and reduce the Iris data set dimensionality to two components
# Apply PCA and reduce the dataset to 2 components
pca = PCA(n_componenets = 2 )
X_pca = pca.fit_transform(X_scaled)

# Plot the PCA-transformed data in 2D
plt.figure(figsize=(8,6))

colors = ['navy', 'turquoise', 'darkorange']
lw = 1
# target_names = ['setosa', 'versicolor', 'virginica']
# y = iris.target
for color, i, target_name in zip(colors, [0, 1, 2], target_names): 
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, s=50, ec='k',alpha=0.7, lw=lw, label=target_name)

plt.title('PCA 2-dimensional reduction of IRIS dataset',)
plt.xlabel("PC1",)
plt.ylabel("PC2",)
plt.legend(loc='best', shadow=False, scatterpoints=1,)
# plt.grid(True)
plt.show()

# Exercise 6. What percentage of the original feature space variance do these two combined principal components explain?
# hint - add the individual values
100*pca.explained_variance_ratio_.sum() # np.float64(95.81320720000164)

"""
A deeper look at the explained variances
In this next and final set of exercises, your goal is to:

Acquire and plot the PCA-explained variance ratios for all four Iris features as a barplot
Overlay the cummulative explained variance

Exercise 7. Reinitialize the PCA model without reducing the dimension
Standardize the Iris data, and fit and transform the scaled data.
"""
# Standardize the data
scalar = StandardScalar()
X_scaled = scalar.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Plot explained variance ratio for each component
plt.figure(figsize=(10,6))
plt.bar(x=range(1, len(explained_variance_ratio)+1), height=explained_variance_ratio, alpha=1, align='center', label='PC explained variance ratio' )
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title('Explained Variance by Principal Components')

# Plot cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)
plt.step(range(1, 5), cumulative_variance, where='mid', linestyle='--', lw=3,color='red', label='Cumulative Explained Variance')
# Only display integer ticks on the x-axis
plt.xticks(range(1, 5))
plt.legend()
plt.grid(True)
plt.show()





 
