# [Import packages]
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import os, random, datetime, re, time, requests, json

# Cluster analysis
# from sklearn.cluster import KMeans  # linear methods
# from sklearn.cluster import SpectralClustering  # nearest neighbors
# from sklearn.datasets import make_moons
# from sklearn.decomposition import PCA  # path collection
from sklearn.metrics import pairwise_distances  # for K means

##############################################################################
#                      Importing self-made functions                         #
##############################################################################
from CPH.scraping import scrape_bolighed
from CPH.cleaning import datastructuring

%matplotlib inline
## plot styles
sns.set_style('ticks')
plt.style.use('seaborn-ticks')

##############################################################################
#                    Scraping the data from bolighed.dk                      #
##############################################################################
# OBS ONLY RUN THIS ONCE. It takes around 18 minuttes.
# !pip3 install tqdm
raw_data = scrape_bolighed(5.0)  # ONLY RUN ONCE. Set sleep time to at least 5 seconds
raw_data = pd.DataFrame(raw_data)
raw_data.to_csv('CPH/Data/raw_data.csv', index=False)  # Save scraped data

##############################################################################
#                               Data cleaning                                #
##############################################################################
# !pip3 install tqdm
# !pip3 install geopy
raw_data = pd.read_csv('CPH/Data/raw_data.csv')
cph = datastructuring(raw_data)
cph.to_csv('CPH/Data/cph.csv', index=False)  # Save scraped data

raw_data = pd.read_csv('https://raw.githubusercontent.com/thornoe/sds_2018/master/CPH/Data/raw_data.csv')
raw_data.head()



##############################################################################
#              Implementing the K-means Clustering algorithm                 #
##############################################################################
cph = pd.read_csv('CPH/Data/cph.csv')
cph.isnull().sum()
