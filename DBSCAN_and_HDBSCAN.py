"""
Comparing DBSCAN and HDBSCAN clustering

Introduction
In this lab, you'll create two clustering models using data curated by StatCan containing the names, types, and locations of cultural and art facilities across Canada.
We'll focus on the museum locations provided across Canada.

Data source: The Open Database of Cultural and Art Facilities (ODCAF)
A collection of open data containing the names, types, and locations of cultural and art facilities across Canada. It is released under the Open Government License - Canada.
The different types of facilities are labeled under 'ODCAF_Facility_Type'.
"""

!pip install numpy==2.2.0
!pip install pandas==2.2.3
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3
!pip install hdbscan==0.8.40
!pip install geopandas==1.0.1
!pip install contextily==1.6.2
!pip install shapely==2.0.6

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.preprocessing import StandardScaler

# geographical tools
import geopandas as gpd  # pandas dataframe-like geodataframes for geographical data
import contextily as ctx  # used for obtianing a basemap of Canada
from shapely.geometry import Point

import warnings
warnings.filterwarnings('ignore')

"""
Download the Canada map for reference
To get a proper context of the final output of this lab, you need a reference map of Canada. Execute the cell below to extract the same to this lab environment.
"""
import requests
import zipfile
import io
import os

# URL of the ZIP file on the cloud server
zip_file_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YcUk-ytgrPkmvZAh5bf7zA/Canada.zip'

# Directory to save the extracted TIFF file
output_dir = './'
os.makedirs(output_dir, exist_ok=True)

# Step 1: Download the ZIP file
response = requests.get(zip_file_url)
response.raise_for_status()  # Ensure the request was successful
# Step 2: Open the ZIP file in memory
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    # Step 3: Iterate over the files in the ZIP
    for file_name in zip_ref.namelist():
        if file_name.endswith('.tif'):  # Check if it's a TIFF file
            # Step 4: Extract the TIFF file
            zip_ref.extract(file_name, output_dir)
            print(f"Downloaded and extracted: {file_name}")

"""
Include a plotting function
The code for a helper function is provided to help you plot your results. Although you don't need to worry about the details, it's quite instructive as it uses a geopandas
dataframe and a basemap to plot coloured cluster points on a map of Canada.
"""
# Write a function that plots clustered locations and overlays them on a basemap.

def plot_clustered_locations(df,  title='Museums Clustered by Proximity'):
    """
    Plots clustered locations and overlays on a basemap.
    
    Parameters:
    - df: DataFrame containing 'Latitude', 'Longitude', and 'Cluster' columns
    - title: str, title of the plot
    """
    
    # Load the coordinates intto a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']), crs="EPSG:4326")
    
    # Reproject to Web Mercator to align with basemap 
    gdf = gdf.to_crs(epsg=3857)
  
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 10))

    # Separate non-noise, or clustered points from noise, or unclustered points
    non_noise = gdf[gdf['Cluster'] != -1]
    noise = gdf[gdf['Cluster'] == -1]
    
    # Plot noise points 
    noise.plot(ax=ax, color='k', markersize=30, ec='r', alpha=1, label='Noise')
    
    # Plot clustered points, colured by 'Cluster' number
    non_noise.plot(ax=ax, column='Cluster', cmap='tab10', markersize=30, ec='k', legend=False, alpha=0.6)
    
    # Add basemap of  Canada
    ctx.add_basemap(ax, source='./Canada.tif', zoom=4)
    
    # Format plot
    plt.title(title, )
    plt.xlabel('Longitude', )
    plt.ylabel('Latitude', )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    
    # Show the plot
    plt.show()

"""
Explore the data and extract what you need from it
Start by loading the data set into a Pandas DataFrame and displaying the first few rows.
"""
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/r-maSj5Yegvw2sJraT15FA/ODCAF-v1-0.csv'
df = pd.read_csv(url, encoding = "ISO-8859-1")
df.head()

"""
Exercise 1. Explore the table. What do missing values look like in this data set?
Strings consisting of two dots '..' indicate missing values. There miight still be empty fields, or NaNs.
"""

"""
Exercise 2. Display the facility types and their counts.
"""
df.ODCAF_Facility_Type.value_counts()
"""
ODCAF_Facility_Type
library or archives                     3013
museum                                  1938
gallery                                  810
heritage or historic site                620
theatre/performance and concert hall     583
festival site                            346
miscellaneous                            343
art or cultural centre                   225
artist                                    94
Name: count, dtype: int64
"""

"""
Exercise 3. Filter the data to only include museums.
"""
df = df[df.ODCAF_Facility_Type == "museum"]   # only the rows that have ODCAF_Facility_Type = museum 
df.ODCAF_Facility_Type.value_counts()
"""
ODCAF_Facility_Type
museum    1938
Name: count, dtype: int64
"""

"""
Exercise 4. Select only the Latitude and Longitude features as inputs to our clustering problem.¶
Also, display information about the coordinates like counts and data types.
**Further downsize the dataframe whose facility type is only "museum"
"""
df = df[['Latitude', 'Longitude']]
df.info()
df.head()
"""
<class 'pandas.core.frame.DataFrame'>
Index: 1938 entries, 1 to 7969
Data columns (total 2 columns):
 #   Column     Non-Null Count  Dtype 
---  ------     --------------  ----- 
 0   Latitude   1938 non-null   object
 1   Longitude  1938 non-null   object
dtypes: object(2)
memory usage: 45.4+ KB

	Latitude	Longitude
1	55.2645508	-127.6428124
2	45.963283	-66.6419017
8	49.1763542	-123.112783
13	49.261938	-123.151123
15	49.88955855	-97.23574396
"""

"""
Exercise 5. We'll need these coordinates to be floats, not objects.
Remove any museums that don't have coordinates, and convert the remaining coordinates to floats.
"""
# Remove observations with no coordinates 
df = df[df.Latitude != ".." ]

# Convert to float
df[['Latitude', 'Longitude']] = df[['Latitude', 'Longitude']].astype('float')
df.info()
"""
<class 'pandas.core.frame.DataFrame'>
Index: 1607 entries, 1 to 7969
Data columns (total 2 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   Latitude   1607 non-null   float64
 1   Longitude  1607 non-null   float64
dtypes: float64(2)
memory usage: 37.7 KB
"""
# --------------------------------------------------------------------------------------------------------------------------------
"""
Build a DBSCAN model
Correctly scale the coordinates for DBSCAN (since DBSCAN is sensitive to scale)
""""

# In this case we know how to scale the coordinates. Using standardization would be an error becaues we aren't using the full range of the lat/lng coordinates.
# Since longitude has a range of +/- 90 degrees and latitude ranges from 0 to 360 degrees, the correct scaling is to double the latitude coordinates (or half the longitudes)
coords_scaled = df.copy()
coords_scaled["Latitude"] = 2 * coords_scaled["Latitude"]

"""
Apply DBSCAN with Euclidean distance to the scaled coordinates¶
In this case, reasonable neighbourhood parameters are already chosen for you. Feel free to experiment.
"""
min_samples = 3 # minimum number of samples needed to form a neighbourhood
eps = 1.0 # neighbourhood search radius
metric = 'euclidean'  # distance measure

dbscan = DBSCAN(eps = eps, min_samples = min_samples, metric = metric).fit(coords_scaled) 

"""
Add cluster labels to the DataFrame
"""
df['Cluster'] = dbscan.fit_predict(coords_scaled) # here we are creating a new column

print(df['Cluster'].value_counts())
df.head()

"""
Cluster
 4     701
 2     192
 1     181
 7     134
 3      94
-1      79
 6      30
 10     27
 8      21
 11     15
 ...
 25      3
 29      3
 31      3
 30      3
 32      3
 As you can see, there are two relatively large clusters and 79 points labelled as noise (-1).
and

	Latitude	Longitude	Cluster
1	55.264551	-127.642812	0
2	45.963283	-66.641902	1
8	49.176354	-123.112783	2
13	49.261938	-123.151123	2
15	49.889559	-97.235744	3
"""

# Plot the museums on a basemap of Canada, colored by cluster label.
plot_clustered_locations(df, title='Museums Clustered by Proximity')

"""
One key thing to notice here is that the clusters are not uniformly dense.

For example, the points are quite densely packed in a few regions but are relatively sparse in between.

DBSCAN agglomerates neighboring clusters together when they are close enough.

Let's see how a hierarchical density-based clustering algorithm like HDBSCAN performs.
"""

"""
Build an HDBSCAN clustering model¶
At this stage, you've already loaded your data and extracted the museum coordinates into a dataframe, df.

You've also stored properly scaled coordinates as the 'coords_scaled' array.

All that remains is to:

Fit and transform HDBSCAN to your scaled coordinates
Extract the cluster labels
Plot the results on the same basemap as before
Reasonable HDBSCAN parameters have been selected for you to start with.
"""

# Initialize an HDBSCAN model
min_samples = None
min_cluster_size = 3
hdb = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size = min_cluster_size, metric='euclidean')  # SEE "import hdbscan" above, also You can adjust parameters as needed

# Exercise 6. Assign the cluster labels to your unscaled coordinate dataframe and display the counts of each cluster label.
# Assign labels
df['Cluster'] = hdb.fit_predict(coords_scaled)  # Another way to assign the labels
# Display the size of each cluster
print(df['Cluster'].value_counts())
df.head()

"""
Cluster
-1      468
 142     45
 95      39
 59      34
 77      29
       ... 
 110      3
 0        3
 133      3
 111      3
 136      3
Name: count, Length: 144, dtype: int64
Latitude	Longitude	Cluster
1	55.264551	-127.642812	4
2	45.963283	-66.641902	33
8	49.176354	-123.112783	-1
13	49.261938	-123.151123	-1
15	49.889559	-97.235744	100

As you can see, unlike the case for DBSCAN, clusters quite uniformly sized, although there is a quite lot of noise identified.
"""
# Exercise 7. Plot the hierarchically clustered museums on a basemap of Canada, colored by cluster label.
plot_clustered_locations(df , title='Museums Hierarchically Clustered by Proximity')
