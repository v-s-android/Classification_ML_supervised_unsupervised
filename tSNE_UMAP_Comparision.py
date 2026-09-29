"""
Objectives¶
After completing this lab you will be able to:

Apply tSNE and UMAP to feature space dimensionality reduction problems
Use PCA as a baseline comparison for evaluating tSNE and UMAP results
"""

"""
Introduction
In this lab, you will explore how to implement two advanced dimensionality reduction algorithms, tSNE and UMAP, on synthetic data. You'll compare the results to the same dimension reduction using PCA.

You'll start by generating a synthetic dataset of blobs in a 3D feature space and visually explore the data in an interactive 3D plot.
Then, you'll use the three algorithms to project the blobs into two dimensions.
For illustrative purposes, you'll color the blobs so we can see what effect the dimension reduction algorithms have on them: how well they preserve structure,
such as the separation between blobs and their relative density.
"""
!pip install numpy==2.2.0
!pip install pandas==2.2.3
!pip install matplotlib==3.9.3
!pip install plotly==5.24.1
!pip install --upgrade scikit-learn umap-learn

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import umap.umap_ as UMAP 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import plotly.express as px

# Generate synthetic data with four clusters in a 3D space

# Custer centers
centers = [ [ 2, -6, -6],
            [-1,  9,  4],
            [-8,  7,  2],
            [ 4,  7,  9] ] 

# Cluster standard deviations:
cluster_std = [1,1,2,3.5]

# Make the blobs and return the data and the blob labels
X, labels_ = make_blobs(centers = centers, cluster_std = cluster_std, n_samples = 500, n_features = 3, random_state = 42 )

# Display the data in an interactive Plotly 3D scatter plot
# Create a DataFrame for Plotly
df = pd.DataFrame(X, columns=['X', 'Y', 'Z'])

# Create interactive 3D scatter plot
fig = px.scatter_3d(df, x='X', y='Y', z='Z', color=labels_.astype(str) ,  opacity=0.7,  color_discrete_sequence=px.colors.qualitative.G10, title="3D Scatter Plot of Four Blobs")

fig.update_traces(marker=dict(size=5, line=dict(width=1, color='black')), showlegend=False)
fig.update_layout(coloraxis_showscale=False, width=1000, height=800)  # Remove color bar, resize plot

fig.show()

"""
Exercise 1. What can you say about the four blobs?

- The blobs have varying densities.
- One blob is distinct from the others.
- The two largest blobs are distinct from each other, but both have a bit of overlap with the other blob between them.
"""

"""
Exercise 2. Standardize the data to prepare it for the three projection methods.
"""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""
Apply t-SNE to reduce the dimensionality to 2D
You'll set the perplexity to the default value of 30 here. The results vary quite a bit if you change the perplexity, so go ahead and experiment.
"""

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

# Let's plot the 2D t-SNE result
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_, cmap='viridis', s=50, alpha=0.7, edgecolor='k')
ax.set_title("2D t-SNE Projection of 3D Data")
ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")
ax.set_xticks([])
ax.set_yticks([])
plt.show()
