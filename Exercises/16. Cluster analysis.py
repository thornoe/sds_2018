import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random, requests

# Cluster analysis
from sklearn.cluster import KMeans  # linear methods
from sklearn.cluster import SpectralClustering  # nearest neighbors
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA  # path collection
from sklearn.metrics import pairwise_distances  # for K means

import plotly.offline as py  # import plotly in offline mode
py.init_notebook_mode(connected=True)  # initialize the offline mode, with access to the internet or not.
import plotly.tools as tls
tls.embed('https://plot.ly/~cufflinks/8')  # conda install -c conda-forge cufflinks-py
# import cufflinks and make it offline
import cufflinks as cf
cf.go_offline()  # initialize cufflinks in offline mode

# First time you run plotly
# import plotly
# username = 'thornoe' # your.username
# api_key = '••••••••••' # find it under settings # your.apikey
# plotly.tools.set_credentials_file(username=username, api_key=api_key)

%matplotlib inline

# Data
sns.set(style="ticks")  # for scatterplot
df = sns.load_dataset("iris")
sns.pairplot(df, hue="species")

### Transformations to lower dimensional space that preserves the variance
plt.style.use('ggplot')
X, y = make_moons(200, noise=.05, random_state=00)
plt.scatter(*zip(*X),c=y,cmap=plt.cm.viridis)
plt.title('True Nonlinear Clusters')

### Standard Linear Approaches.
labels = KMeans(2, random_state=0).fit_predict(X)
fig, axes = plt.subplots(1,2)
axes[0].scatter(X[:, 0], X[:, 1], c=labels,
            s=50, cmap='viridis');

pca = PCA(n_components=1)
x = pca.fit_transform(X)
axes[1].scatter(x,y,c=y) #

### Nearest neighbors
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                           assign_labels='kmeans')
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels,
            s=50, cmap='viridis');

### TSNE: t-distributed stochastic neighbor embedding
# - Widely used algorithm for exploring highdimensional data.
# - Problem: lacks interpretable global properties.

### DOESN'T WORK - CAN'T BE FOUND ###
url1 = 'https://raw.githubusercontent.com/snorreralund/sds_dump/master/w2vec_dimensionality_reduction.npy'
response1 = requests.get(url1)
response1
type(response1)
w2vec_file = np.load(response1)

url2 = 'https://raw.githubusercontent.com/snorreralund/sds_dump/master/tfidf_dimensionality_reduction.npy'
response2 = requests.get(url2)
response2
tfidf_file = np.load(response2)
# pca_embedding_idf, tsne_embedding_idf, umap_embedding_idf = np.load('tfidf_dimensionality_reduction.npy')
# pca_embedding, tsne_embedded, umap_embedding = np.load('w2vec_response')

### UMAP: Uniform Manifold Approximation and Projection (has global properties)
df = pd.read_csv('https://raw.githubusercontent.com/snorreralund/scraping_seminar/master/english_review_sample.csv')

def build_hovertext(row):
    """function for constructing the hovertext combining the rating value and the review body.
    Further more it Replaces standard newline character \n with html newline tag <br>"""
    return 'Rating:%d <br><br>Review:%s'%(row['reviewRating_ratingValue'],'<br>'.join(row['reviewBody'].split('\n')))
def plot_embedding(df,embedding,title,n=1000,max_string=500,interactive=False):
    "Helper function (i.e. not a general purpose function) for plotting the data."
    df['x'] = embedding[:,0]
    df['y'] = embedding[:,1]
    df['Sentiment'] = df['reviewRating_ratingValue'].apply(lambda x: 'Positive' if x>3 else 'Negative')
    df = df.groupby('reviewRating_ratingValue').apply(lambda x:x.sample(100))
    # df = make_color_scale(df,'reviewRating_ratingValue')
    df['text'] = df.apply(build_hovertext, axis=1)
    if interactive:
        df.iplot(kind='scatter',x='x',y='y',categories='reviewRating_ratingValue',text='text',title=title,colorscale='OrRd')  # ,layout=layout)
    else:
        sns.lmplot(x='x',y='y',hue='Sentiment',data=df,fit_reg=False,scatter_kws={'alpha':0.3})
        plt.title(title)

embeddings = {'pca':pca_embedding,'pca_tfidf':pca_embedding_idf
                  ,'tsne':tsne_embedded,'tsne_tfidf':tsne_embedding_idf,
                 'umap':umap_embedding,'umap_tfidf':umap_embedding_idf}
for name,embedding in enumerate(sorted(embeddings.items(),key=lambda x: len(x[0]))):
    if not '_' in name:
        name = name+'_w2vec'
    plot_embedding(df,embedding,name,interactive=False)

plot_embedding(df,umap_embedding,'umap_embedding_w2vec',interactive=True)

layout = {'autosize': False,
    'width': 900,
    'height': 600}  # fix problem when converting to slides
##############################################################################
#          Ex. 16.2: Implementing the K-means Clustering algorithm           #
##############################################################################
sns.set(style="ticks")  # for scatterplot
df = sns.load_dataset('iris')
sns.pairplot(df, hue='species')
len(df)

### 16.2.1 Define a matrix using the .values method
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
