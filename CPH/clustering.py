# [Import packages]
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, random, datetime, re, time, requests, json
# import tqdm  # !pip3 install tqdm #

# GeoPy - see https://pypi.org/project/geopy/
# !pip3 install geopy
from geopy.geocoders import Nominatim  # retrieve coordinates from adresses etc.
from geopy.extra.rate_limiter import RateLimiter  # delay between calls

# Cluster analysis
# from sklearn.cluster import KMeans  # linear methods
# from sklearn.cluster import SpectralClustering  # nearest neighbors
# from sklearn.datasets import make_moons
# from sklearn.decomposition import PCA  # path collection
from sklearn.metrics import pairwise_distances  # for K means

%matplotlib inline
## plot styles
sns.set_style('ticks')
plt.style.use('seaborn-ticks')

##############################################################################
#              Implementing the K-means Clustering algorithm                 #
##############################################################################
cph_data = pd.read_csv('Project/Data/cph_data_sorted.csv')
cph_data.tail(2)
df = cph_data.loc[:,['Latitude', 'Longitude', 'Sqm_price']].copy


sns.pairplot(df, hue='species')
len(df)

### Define a matrix using the .values method
X = df.iloc[:,:4].values

### 16.2.2
def initialize_clusters(k,X):
    """Initialization: first Expectation of the cluster centroids (centers)"""
    idx = random.sample(range(len(X)),k)
    centroids = X[idx]
    return centroids
k = 3
centroids = initialize_clusters(k,X)
centroids

### 16.2.3 Maximization step - assign each datapoint to the closests centroids
def maximize(centroids,X):
    """Calculates the distance from each point to each centroid (cluster center)
    Next runs an argmin operation on the matrix to obtain the cluster_assignment,
    assigning each datapoint to the closest centroid."""
    dist_matrix = pairwise_distances(centroids,X)
    cluster_assignment = dist_matrix.T.argsort(axis=1)[:,0]
    return cluster_assignment
cluster_assignment = maximize(centroids,X)
cluster_assignment

### 16.2.4 Updating expectation step
def update_expectation(k,X,cluster_assignment):
    """Update Expectation of the cluster centroids by applying the .mean function
    on the subset of the data that is assigned to each cluster"""
    new_centroids = np.zeros((k,len(X[0])))
    for i in range(k):
        subset = X[cluster_assignment==i]  # filter the data with a boolean vector
        new_centroids[i] = subset.mean(axis=0)  # calculate the mean on the subset (axis=0 is on the columns)
    return new_centroids

### 16.2.5 Convergence
def fit_transform(k,X,max_iter):
    old_centroids = initialize_clusters(k,X)
    iterations = 0
    for i in np.arange(max_iter):
        cluster_assignment = maximize(old_centroids,X)
        new_centroids = update_expectation(k,X, cluster_assignment)
        if np.array_equal(old_centroids, new_centroids):
            break
        else:
            old_centroids = new_centroids
            iterations = iterations + 1
    return cluster_assignment, old_centroids, iterations

max_iter = 15  # maximum number of iterations
k = 3
cluster_assignment, centroids, iterations = fit_transform(k, X, max_iter)
print('Cluster assignment:\n', cluster_assignment,
    '\nCentroids:\n', centroids,
    '\nNumber of iterations:\n', iterations)

### 16.2.6 Count overlap between species and clusters
df['cluster'] = cluster_assignment
pd.pivot_table(df,'cluster','species',aggfunc='count')
