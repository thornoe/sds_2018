import warnings
import numpy as np
import pandas as pd

# Other packages
import os
import matplotlib.pyplot as plt
import requests
import json
import seaborn as sns
import pprint
import re
import sys

sys.modules

%matplotlib inline
# Changing default plots
plt.style.use('default')  # set style (colors, background, size, gridlines etc.)
plt.rcParams['figure.figsize'] = 6, 4  # set default size of plots
plt.rcParams.update({'font.size': 10})
sns.set(style='ticks', palette="RdBu")
# for item in ax.get_yticklabels()+ax.get_xticklabels():
#     item.set_fontsize(12)


# Data

############################################
#   6. Data structuring - the Pandas way   #
############################################
str_ser1 = pd.Series(['Andreas','Snorre','Ulf'])
str_ser1 + ' works @ SODAS'
str_ser2 = pd.Series(['Bjerre-Nielsen', 'Ralund', 'Aslak'])
str_ser1 + str_ser2
# String operations vectorized  #
# The powerful .str has several powerful methods e.g. `contains`, `capitalize`. Example:
str_ser1.str.upper()
str_ser1.str.contains('D')
str_ser2.str[0]
####################################
#   Categorical data vs. string    #
####################################
# Storage and faster computation (sometimes)
# Allows for ordering strings
edu_list = (['B.Sc. Political Science', 'Secondary school'] + ['High school']*2)*3
edu_cats = ['Secondary school', 'High school', 'B.Sc. Political Science']  # Specify order
str_ser3 = pd.Series(edu_list)

# option 1 - when ordering matters
cats = pd.Categorical(str_ser3, categories=edu_cats, ordered=True)
cat_ser = pd.Series(cats, index=str_ser3)

# option 2 - no order - fast
cat_ser2 = str_ser3.astype('category')
####################################
#           Temporal data          #
####################################
str_ser4 = pd.Series(['20170101', '20170727', '20170803', '20171224'])
dt_ser1 = pd.to_datetime(str_ser4)
dt_ser1
print(dt_ser1.astype(np.int64))  # epoch time
    # nano seconds from 1970, 00:00 GMT (don't convert!)

## Plot time series
T = 1000
data = {v: np.cumsum(np.random.randn(T)) for v in ['A', 'B']}
data['time'] = pd.date_range(start='20150101', freq='D', periods=T)
ts_df = pd.DataFrame(data)

ts_df.set_index('time').plot()  # Set to 'time' index

# Extract the date
dt_ser2 = ts_df.time
dt_ser2.dt.day.head(3)
dt_ser2.dt.year.head(3)
dt_ser2.dt.month.iloc[500:505]
####################################
#           Missing data           #
####################################
nan_data = [[1,np.nan,3],
            [4,5,None],
            [7,8,9]]
nan_df = pd.DataFrame(nan_data, columns=['A', 'B', 'C'])
nan_df
nan_df.isnull()  # converts to boolean dataframe (is missing?)
nan_df.isnull().sum()
#                  Handling missing data                   #
# 1. Ignore the problem
# 2. Drop missing data: columns and/or rows
# 3. Fill in the blanks
# 4. If time and money permits: collect the data or new data

# Using the 'dropna' method
nan_df.dropna()  # remove all rows w. missing data
nan_df.dropna(subset=['B'])  # remove missing in 'B' column only
nan_df.dropna(axis=1)   # remove all columns w. missing data

# Filling missing data
nan_df.fillna(0)
selection = nan_df.B.isnull()
nan_df.loc[selection, 'B'] = -99  # Replacing in column 'B' only
nan_df

# Duplicates in data
str_ser3
str_ser3.drop_duplicates()  # Keeping 1st instance

#               Binning numerical data - based on quantiles            #
# cut which divides data by user specified bins
# qcut which divides data by user specified quantiles (e.g. median, q=0.5)
x = pd.Series(np.random.normal(size=10**6))
cat_ser3 = pd.qcut(x, q=[0, .95, 1])
cat_ser3.cat.categories
# we can create dummy variables from categorical with to_dummies
# we can combine many of the tools we have learned with groupby (tomorrow)

#############################################################################
#                   Exercises Set 6: Data structuring 2                     #
#############################################################################
## 6.1.1 restore weather data
url_w = 'https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/1864.csv.gz'
df1 = pd.read_csv(url_w, header=None, compression='gzip')
# Label/remove variables
df2 = df1.iloc[:, :4]
df2.columns = ['identifier', 'date', 'type', 'value']
df2['value'] = df2['value'] / 10
df2.head(3)
# Only rows for max-temperature
df2_maxt = df2['type'] == 'TMAX'
df_w = df2[df2_maxt].copy()

# reset index
df_w = df_w.reset_index(drop=True)

## 6.1.2 convert to date
date_n = pd.to_datetime(df_w['date'], format='%Y%m%d', errors='ignore')
df_w['date'] = date_n

## 6.1.3 set as temporal index and plot
df_w = df_w.set_index('date')
# time series plot
df_w['value'].plot(x='date', title='Max temperature').set_ylabel('Degrees celsius')
df_w.head(3)
## 6.1.4 extract the country code from the station name into a separate column
df_w['country'] = df_w['identifier'].str[0:2]
## 6.1.4 create a function that does all of the above
def weather_for_year(year):
    url = 'https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/'+str(year)+'.csv.gz'
    df1 = pd.read_csv(url,
                         compression='gzip',
                         header=None).iloc[:,:4]
    df1.columns = ['station', 'date', 'type', 'value']
    maxt = df1['type'] == 'TMAX'
    df_w = df1[maxt].copy()  # Only max temperatur
    df_w['value'] = df_w['value'] / 10
    df_time = df_w['date'].copy()
    df_time = pd.to_datetime(df_time, format='%Y%m%d')
    df_w['month'] = df_time.dt.month
    df_w['year'] = df_time.dt.year
    df_w['date'] = df_time
    df_w.set_index(df_time, inplace=True)
    df_w.drop(columns='date')
    df_w['country'] = df_w['station'].str[0:2]
    print('df_w created as follows')
    print(df_w.head(3))

weather_for_year(1864)
df_w.head(3)
df_w.drop(columns=['identifier'])


## 4.1.7 Save using relative path
df_w.to_csv('Exercises/weather1864.csv', index=False)  # default is to save the index
