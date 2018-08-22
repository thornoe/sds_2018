import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cluster analysis
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans  # linear methods
from sklearn.decomposition import PCA  # path collection
from sklearn.cluster import SpectralClustering  # nearest neighbors

%matplotlib inline
# Plots
sns.set(style="ticks")
plt.style.use('ggplot')

# Data
df = sns.load_dataset("iris")
sns.pairplot(df, hue="species")

### Transformations to lower dimensional space that preserves the variance
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
### UMAP: Uniform Manifold Approximation and Projection (has global properties)
