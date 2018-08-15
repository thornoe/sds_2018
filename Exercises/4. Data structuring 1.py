import numpy as np
import pandas as pd

# Other packages
import os
import matplotlib.pyplot as plt
import requests
import json
import seaborn as sns
import pprint

# Slides
df = pd.DataFrame([[1,2], [3, 4]],
    columns=['A', 'B'],
    index=['i', 'ii'])

my_series = pd.Series([1,True,1/3,'a'])
my_serie

to.dict ??

df.A
 df['B']

# Ex
num_data = range(0,3)
indices = ['A', 'C', 'B']
my_series2 = pd.Series(num_data, index=indices)  # indices are like the keys
my_series2

# Converting
my_series2.astype(np.float64) # np.str
# Numeric operations
my_arr1 = np.array([2,3,2])
my_arr2 = my_arr1 ** 2
my_arr2
pd.Series(my_arr2)//2
arr = np.random.normal(size=10**7)
s2 = pd.Series(arr)
s2.median()
s2.describe()
my_series4 = pd.Series(my_arr2)
my_series4.unique()
my_series4.value_counts()
# Other numeric methods
unique, nunique # the unique elements and the count of unique elements
cut, qcut # partition series into bins
diff # difference every two consecutive observations
cumsum # cumulative sum
nlargest, nsmallest # the n largest elements
idxmin, idxmax # index which is minimal/maximal
corr # correlation matrix

help(pd.Series.value_counts)
##################
# Boolean series #
##################
my_series2 > 0
my_rng = range(2)
list(my_rng)
my_series2.isin(my_rng)
((my_series2>0) & (my_series2==1))

my_series2[my_series2<2]  # boolean series within a series

df=
(my_df.A > 2) & my_df.B<4)
#########################################
# Inspecting and selecting observations #
#########################################
n = 3  # number of observations
arr = np.random.normal(size=[10*10**6])
my_series7 = pd.Series(arr)
my_series7  # prints "head and tail" of the dataset

my_loc = ['A','C','B']  # Select rows given index keys
my_series2[my_loc]

my_series2[my_loc]
my_series2.iloc[1:3]  # Select rows given index integers

arr[90000:90003]

# loc and iloc for DataFrame
my_idx = ['i', 'ii', 'iii']
my_cols = ['a','b']
my_data = [[1,2], [3,4], [5, 6]]
my_df = pd.DataFrame(my_data, columns=my_cols, index=my_idx)
my_df.loc[['i']]

idx_keep = ['i','ii']
cols_keep = ['a']
my_df.loc[idx_keep, cols_keep]
########################
# Modifying DataFrames #
########################
my_df.set_index('a')  # Doesn't change the DF
my_df_a = my_df.set_index('a')
my_df_copy = my_df.copy()
my_df_copy.set_index('a', inplace=True)  # explicitly replaces
my_df_copy

my_df.reset_index(drop=True)
my_df
#########################
# Changing column value #
#########################
my_df['B'] = [2,17,0]  # set different values
# using loc iloc

my_df,sort_values(by='a', ascending=False)
######################
# Reading DataFrames #
######################
url = 'https://api.statbank.dk/v1/data/FOLK1A/CSV?lang=en&Tid=*'
abs_path = 'C:=....'
rel_path = 'FOLK1A.csv'
df = pd.read_csv(url, sep =';')  # semicolon separated
df.head(3)
df.tail(3)
df.to_csv('DST_people_count.csv', index=False)
#####################
#     EXERCISES     #
#####################
### Ex 4.1 Weather, part 1
## Ex. 4.1.1
url_w = 'https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/1864.csv.gz'
df1 = pd.read_csv(url_w, header=None, compression='gzip')
df1.head(10)
## Ex. 4.1.2
df2 = df1.iloc[:, :4]
df2.columns = ['identifier', 'date', 'type', 'value']
df2['value'] = df2['value'] / 10

## Ex. 4.1.3
df2_station = df2['identifier'] == 'ITE00100550'
df2_maxt = df2['type'] == 'TMAX'
df2_boolean = df2_station & df2_maxt
df3 = df2[df2_boolean].copy()

## Ex. 4.1.4
df3b = df3['value']*1.8 + 32
df3b = pd.DataFrame(df3b)
df3b.columns = ['fahrenheit']
df3b.count()
df4 = pd.concat([df3, df3b], axis=1, sort=True)
type(df3b)
df4.tail(3)
## 4.1.5




help(pd.DataFrame.columns)
