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
X_tsne
"""
array([[ 5.54592276e+00,  1.77139034e+01],
       [ 6.99924648e-01, -1.96951275e+01],
       [ 1.04250975e+01,  1.85730438e+01],
       [ 6.06361961e+00, -2.44908657e+01],
       ...
"""

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

"""
Exercise 3. What can you say about this t-SNE result?

- t-SNE projected the data into four distinct clusters, although the original data had some overlap between a few clusters.
- You can see that some of the points ended up in the "wrong" cluster, although to be fair, t-SNE has no knowledge of which clusters the points actually belong to.
- All the clusters have similar densities.
- Two of the blobs are distinct from each other but "gave up" some of their points to the blob they originally had overlapped with.
- A "perfect" result would not completely separate the overlaps between blobs.
- Notice that the distance between the blobs is consistent with the degree to which they were originally separated.
"""

# Compare UMAP and PCA dimensionality reduction to two dimensions
# Apply UMAP to reduce the dimensionality to 2D
umap_model = UMAP.UMAP(n_components = 2, random_state = 42, min_dist = 0.5, spread = 1, n_jobs = 1) # import umap.umap_ as UMAP 
X_umap = umap_model.fit_transform(X_scaled)
print("the shape of X_umap is ",X_umap.shape) # the shape of X_umap is  (500, 2)
"""
- n_components=2 → Reduces your data to 2 dimensions for visualization.
- random_state=42 → Makes the result reproducible.
- min_dist=0.5 → Controls how tightly points can cluster. Smaller = tighter clusters.
- spread=1 → Controls the overall scale/spread of the embedding. 1 is the standard/default.
- n_jobs=1 → Uses 1 CPU thread.

In one sentence: This creates a reproducible 2D UMAP visualization with moderately compact clusters using a single CPU thread.
"""

# plot the 2D UMAP prjection
fig = plt.figure(figsize = (8,6))
ax = fig.add_subplot(111) # Adds a subplot/axes to the figure.
"""
The 111 means:
1 row
1 column
1st subplot
So you're creating one plotting area that occupies the entire figure.
its same as fig.add_subplot(1, 1, 1)
"""

ax.scatter( X_umap[:,0], X_umap[:,1], c=labels_, cmap='viridis', s=50, alpha=0.7, edgecolor='k')

"""
Creates a scatter plot of the 2D UMAP results.

X_umap[:,0] → X-axis
X_umap[:,1] → Y-axis

Point 1 → (X_umap[0,0], X_umap[0,1])
Point 2 → (X_umap[1,0], X_umap[1,1])
Point 3 → (X_umap[2,0], X_umap[2,1])
...

c=labels_ → Color points by their cluster/label
cmap='viridis' → Color scheme
s=50 → Point size
alpha=0.7 → 70% transparency
edgecolor='k' → Black borders around points
"""
ax.set_title("2D UMAP Projection of 3D Data")
ax.set_xlabel("UMAP Component 1", )
ax.set_ylabel("UMAP Component 2", )
ax.set_xticks([])
ax.set_yticks([])
plt.show()

# Exercise 4. What can you say about this UMAP result?
# Apply PCA to reduce the dimensionality to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


fig = plt.figure(figsize=(8, 6))

# Plot the 2D PCA result (right)
ax2 = fig.add_subplot(111)
scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_, cmap='viridis', s=50, alpha=0.7, edgecolor='k')
ax2.set_title("2D PCA Projection of 3-D Data")
ax2.set_xlabel("PCA 1")
ax2.set_ylabel("PCA 2")
ax2.set_xticks([])
ax2.set_yticks([])
plt.show()

"""
Exercise 5. What can you say about this PCA result?

- PCA faithfully preserved the relative blob densities.
- PCA also preserved the relative separation between blobs.
- The distance between the clusters is very consistent with the degree to which they were originally separated.
- PCA and t-SNE took very little time to complete compared to UMAP.
- IMNSHO, PCA outperformed both t-SNE and UMAP in this experiment. This points to a common tendency to want to implement more advanced algorithms.
The default result is not always an improvement over the simpler established methods.
""
