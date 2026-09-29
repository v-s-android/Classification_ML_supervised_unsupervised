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
from sklearn.datasets import make_blobs

# Generate synthetic data with four clusters in a 3D space
