# [Import packages]
import numpy as np
import pandas as pd
import seaborn as sns
# from shapely.geometry import Point
import matplotlib.pyplot as plt
import random  # tqdm, datetime, re, time, requests, json

# Cluster analysis
# from sklearn.cluster import KMeans  # linear methods
# from sklearn.cluster import SpectralClustering  # nearest neighbors
# from sklearn.datasets import make_moons
# from sklearn.decomposition import PCA  # path collection
from sklearn.metrics import pairwise_distances  # for K means

%matplotlib inline

##############################################################################
#         Coding functions for the the K-means Clustering algorithm          #
##############################################################################
def initialize_clusters(X, k, seed):
    """Initialization: first Expectation of the cluster centroids (centers)"""
    random.seed(2900)
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
def fit_transform(X, k, max_iter, seed):
    old_centroids = initialize_clusters(X, k, seed)
    iterations = 0
    for i in np.arange(max_iter):
        cluster_assignment = maximize(X, old_centroids)
        new_centroids = update_expectation(X, k, cluster_assignment)
        if np.array_equal(old_centroids, new_centroids):
            break
        else:
            old_centroids = new_centroids
            iterations = iterations + 1
    print('Centroids:', '\n', old_centroids,
        '\nIterations:', '\n', iterations,)
    return cluster_assignment, old_centroids

##########################################################################
#             Implementing the K-means Clustering algorithm              #
##########################################################################
# RUN THE CODE FOR CLUSTERING
# !pip3 install tqdm
cph = pd.read_csv('CPH/Data/cph.csv')
seed = 2700

##########################################################################
#            Clusters in Greater Copenhagen - 3-dimensional              #
##########################################################################
X = cph.loc[:,['Latitude', 'Longitude', 'log_sqm_price']].values  # Define a matrix using the .values method

max_iter = 100  # maximum number of iterations
k = 3  # number of clusters
cluster_assignment, centroids = fit_transform(X, k, max_iter, seed)  # Set the number of clusters

XD = cph.reindex(columns = ['Latitude', 'Longitude', 'log_sqm_price', 'Sqm_price'])
XD.insert(loc=3, column='Cluster', value=cluster_assignment)
XD['Cluster'] = XD['Cluster'].astype('category')
# XD['Cluster'].dtype
XD['Cluster'] = XD['Cluster'].cat.rename_categories(['High', 'Low', 'Middle'])  # ({'0': 'High', '1':'Low', '2':'Middle'})
XD['Cluster'].cat.categories

centroids = pd.DataFrame(centroids)
centroids.columns = ['Latitude', 'Longitude', 'log_sqm_price']
cluster = ['High', 'Low', 'Middle']
centroids.insert(loc=3, column='Cluster', value=cluster)

# Plot
sns.set(style='ticks')
fig, ax = plt.subplots(figsize = (15, 15*0.87))
ax1 = sns.scatterplot(x='Longitude', y='Latitude', hue='Cluster', hue_order=['Low', 'Middle', 'High'],
    size = 'log_sqm_price', sizes=(1,300), palette="Paired", legend='brief', data=XD, alpha=0.4)
ax2 = sns.scatterplot(x='Longitude', y='Latitude', hue='Cluster', hue_order=['Low', 'Middle', 'High'],
    size = 'log_sqm_price', sizes=(800,1200), palette="Paired", legend=False, data=centroids, marker='X', alpha=0.9)
ax = ax1, ax2
fig.savefig("CPH/Fig/cluster.pdf", dpi=600, bbox_inches='tight')

XD['Cluster'].value_counts(normalize=True)

data_0 = XD['Sqm_price'].loc[XD['Cluster'] == 'Low']
data_1 = data_0.describe([.05, .5, .95])
data_0 = XD['Sqm_price'].loc[XD['Cluster'] == 'Middle']
data_2 = data_0.describe([.05, .5, .95])
data_0 = XD['Sqm_price'].loc[XD['Cluster'] == 'High']
data_3 = data_0.describe([.05, .5, .95])
CPH_desc = pd.DataFrame(data_1)
CPH_desc.columns = ['Low']
CPH_desc.insert(loc=1, column='Middle', value=data_2)
CPH_desc.insert(loc=2, column='High', value=data_3)
# CPH_desc = CPH_desc.round(decimals=0)
CPH_desc = CPH_desc.astype(int)
descriptives = CPH_desc.transpose()
descriptives.to_latex()  # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_latex.html
CPH_desc


# Round of decimal points and choose descriptives of interest.




data_3

# latitude_span = 55.94 - 55.522
# longitude_span = 12.67 - 12.19
# print(latitude_span / longitude_span)
##########################################################################
#        Clusters in Greater Copenhagen - log yearly sqm expenses        #
##########################################################################
Z = cph.loc[:,['Latitude', 'Longitude', 'log_yearly_sqm_exp']].values  # Define a matrix using the .values method

max_iter = 100  # maximum number of iterations
k = 3  # number of clusters
cluster_assignment, centroids = fit_transform(Z, k, max_iter, seed)  # Set the number of clusters

ZD = cph.reindex(columns = ['Latitude', 'Longitude', 'log_yearly_sqm_exp', 'Yearly_expenses'])
ZD.insert(loc=3, column='Cluster', value=cluster_assignment)
ZD['Cluster'] = ZD['Cluster'].astype('category')
# ZD['Cluster'].dtype
ZD['Cluster'] = ZD['Cluster'].cat.rename_categories(['High', 'Middle', 'Low'])  # ({'0': 'High', '1':'Middle', '2':'Low'})
ZD['Cluster'].cat.categories

centroids = pd.DataFrame(centroids)
centroids.columns = ['Latitude', 'Longitude', 'log_yearly_sqm_exp']
cluster = ['High', 'Middle', 'Low']
centroids.insert(loc=3, column='Cluster', value=cluster)

# Plot
sns.set(style='ticks')
fig, ax = plt.subplots(figsize = (15, 15*0.87))
ax1 = sns.scatterplot(x='Longitude', y='Latitude', hue='Cluster', hue_order=['Low', 'Middle', 'High'],
    size = 'log_yearly_sqm_exp', sizes=(1,300), palette="Paired", legend='brief', data=ZD, alpha=0.4)
ax2 = sns.scatterplot(x='Longitude', y='Latitude', hue='Cluster', hue_order=['Low', 'Middle', 'High'],
    size = 'log_yearly_sqm_exp', sizes=(800,1200), palette="Paired", legend=False, data=Zcentroids, marker='X', alpha=0.9)
ax = ax1, ax2
# fig.savefig("CPH/Fig/cluster_yearlyexpenses.pdf", dpi=600, bbox_inches='tight')

ZD['Cluster'].value_counts(normalize=True)

ZD['Differences'] = np.where(XD['Cluster'] != ZD['Cluster'], ZD['Cluster'], np.nan)
XD['Cluster'].value_counts(normalize=True)
ZD['Cluster'].value_counts(normalize=True)
ZD['Differences'].value_counts(normalize=False)
(311+109+32)/len(ZD)  # 11.4 pct. belongs to a different cluster

# cph.loc[:, ['log_sqm_price', 'Sqm_price', 'Price', 'Owner_expense', 'Yearly_expenses', 'First_year_expenses', 'log_yearly_sqm_exp']].head()
cph.loc[:, ['log_sqm_price', 'Sqm_price', 'Price', 'Owner_expense', 'Yearly_expenses', 'First_year_expenses', 'log_yearly_sqm_exp']].describe(percentiles = [.25, .5, .75])
0.282/10.641  # standard deviation for log_sqm_price
0.2183/7.5618 # std.dev. a little higher for log_yearly_sqm_exp
# ZD['Yearly_expenses'].loc[ZD['Cluster'] == 'Low'].describe([.05, .5, .95])
# ZD['Yearly_expenses'].loc[ZD['Cluster'] == 'Middle'].describe([.05, .5, .95])
# ZD['Yearly_expenses'].loc[ZD['Cluster'] == 'High'].describe([.05, .5, .95])
##########################################################################
#                 Clusters in Gentofte - 3-dimensional                   #
##########################################################################
gentofte = cph.loc[cph['Municipality'] == 'Gentofte']
X = gentofte.loc[:,['Latitude', 'Longitude', 'log_log_sqm_price']].values  # Define a matrix using the .values method

max_iter = 100  # maximum number of iterations
k = 3  # number of clusters
cluster_assignment, centroids = fit_transform(X, k, max_iter, seed)  # Set the number of clusters

XD = gentofte.reindex(columns = ['Latitude', 'Longitude', 'log_log_sqm_price'])
XD.insert(loc=3, column='Cluster', value=cluster_assignment)

centroids = pd.DataFrame(centroids)
centroids.columns = ['Latitude', 'Longitude', 'log_log_sqm_price']
num_data = range(0,k)
centroids.insert(loc=3, column='Cluster', value=num_data)

fig, ax = plt.subplots(figsize = (15, 15*0.625))
ax2 = sns.scatterplot(x='Longitude', y='Latitude', hue='Cluster', size = 'log_log_sqm_price', sizes=(300,600), palette="Paired", legend=False, data=centroids, marker='x')
ax1 = sns.scatterplot(x='Longitude', y='Latitude', hue='Cluster', size = 'log_log_sqm_price', sizes=(1,300), palette="Paired", legend=False, data=XD,)
# fig1.set(ylabel='Price per square meter', xlabel='Area', title='Log log_sqm price plottet against area with rooms as hue')

XD.loc[XD['Cluster'] == 0].describe()
XD.loc[XD['Cluster'] == 1].describe()
XD.loc[XD['Cluster'] == 2].describe()

# XD.describe()
# latitude_span = 55.77 - 55.72
# longitude_span = 12.60 - 12.52
# print(latitude_span / longitude_span)
##########################################################################
# Clusters in Gentofte - n-dimensional, n is number of numeric variables #
##########################################################################
X = gentofte.loc[:,['Zip_code', 'Latitude', 'Longitude', 'Floor', 'Rooms',
    'Area', 'Land_area', 'log_log_sqm_price', 'Owner_expense', 'Price_development',
    'Energy_saving', 'Days_on_market']].values  # Define a matrix using the .values method

max_iter = 100  # maximum number of iterations
k = 3  # number of clusters
cluster_assignment, centroids = fit_transform(X, k, max_iter)  # Set the number of clusters

XD = gentofte.reindex(columns = ['Latitude', 'Longitude', 'log_log_sqm_price'])
XD.insert(loc=3, column='Cluster', value=cluster_assignment)

centroids = pd.DataFrame(centroids)
centroids = centroids.iloc[:,[1, 2, 7]]
centroids.columns = ['Latitude', 'Longitude', 'log_log_sqm_price']
num_data = range(0,k)
centroids.insert(loc=3, column='Cluster', value=num_data)

fig, ax = plt.subplots(figsize = (15, 15*0.625))
ax2 = sns.scatterplot(x='Longitude', y='Latitude', hue='Cluster', size = 'log_log_sqm_price', sizes=(300,600), palette="Paired", legend=False, data=centroids, marker='x')
ax1 = sns.scatterplot(x='Longitude', y='Latitude', hue='Cluster', size = 'log_log_sqm_price', sizes=(1,300), palette="Paired", legend=False, data=XD,)
# fig1.set(ylabel='Price per square meter', xlabel='Area', title='Log log_sqm price plottet against area with rooms as hue')
