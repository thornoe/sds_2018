# [Import packages]
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm, random  # os, datetime, re, time, requests, json

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
#         Coding functions for the the K-means Clustering algorithm          #
##############################################################################
def initialize_clusters(X, k):
    """Initialization: first Expectation of the cluster centroids (centers)"""
    idx = random.sample(range(len(X)), k)
    centroids = X[idx]
    return centroids

# Maximization step - assign each datapoint to the closests centroids
def maximize(X, centroids):
    """Calculates the distance from each point to each centroid (cluster center)
    Next runs an argmin operation on the matrix to obtain the cluster_assignment,
    assigning each datapoint to the closest centroid."""
    dist_matrix = pairwise_distances(centroids, X)
    cluster_assignment = dist_matrix.T.argsort(axis=1)[:,0]
    return cluster_assignment

# Updating expectations step
def update_expectation(X, k, cluster_assignment):
    """Update Expectation of the cluster centroids by applying the .mean function
    on the subset of the data that is assigned to each cluster"""
    new_centroids = np.zeros((k,len(X[0])))
    for i in range(k):
        subset = X[cluster_assignment==i]  # filter the data with a boolean vector
        new_centroids[i] = subset.mean(axis=0)  # calculate the mean on the subset (axis=0 is on the columns)
    return new_centroids

# Convergence
def fit_transform(X, k, max_iter):
    old_centroids = initialize_clusters(X, k)
    iterations = 0
    for i in tqdm.tqdm(np.arange(max_iter)):
        cluster_assignment = maximize(X, old_centroids)
        new_centroids = update_expectation(X, k, cluster_assignment)
        if np.array_equal(old_centroids, new_centroids):
            break
        else:
            old_centroids = new_centroids
            iterations = iterations + 1
    print('Iterations:', iterations)
    return cluster_assignment, old_centroids

##########################################################################
#             Implementing the K-means Clustering algorithm              #
##########################################################################
# RUN THE CODE FOR CLUSTERING


##########################################################################
#                         Clusters in Gentofte                          #
##########################################################################
# k = 10  # number of closters
for k in range(2,5):
    cluster_assignment, centroids = fit_transform(X, k, max_iter)  # Set the number of clusters
    XD = pd.DataFrame(X)
    XD.columns = ['Latitude', 'Longitude', 'log_sqm_price']
    XD.insert(loc=3, column='Cluster', value=cluster_assignment)
    centroids = pd.DataFrame(centroids)
    centroids.columns = ['Latitude', 'Longitude', 'log_sqm_price']
    num_range = range(0, k)
    centroids.insert(loc=3, column='Cluster', value=num_range)

    fig, ax = plt.subplots(figsize = (15, 15*0.625))
    ax2 = sns.scatterplot(x='Longitude', y='Latitude', hue='Cluster', size = 'log_sqm_price', sizes=(300,600), palette="Paired", legend=False, data=centroids, marker='x')

    fig, ax = plt.subplots(figsize = (15, 15*0.625))
    ax2 = sns.scatterplot(x='Longitude', y='Latitude', hue='Cluster', size = 'log_sqm_price', sizes=(300,600), palette="Paired", legend=False, data=centroids, marker='x')
    ax1 = sns.scatterplot(x='Longitude', y='Latitude', hue='Cluster', size = 'log_sqm_price', sizes=(1,300), palette="Paired", legend=False, data=XD,)


fig1.set(ylabel='Price per square meter', xlabel='Area', title='Log sqm price plottet against area with rooms as hue')
