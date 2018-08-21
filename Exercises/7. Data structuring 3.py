import warnings
import numpy as np
import pandas as pd
from Exercises.weather import weather_for_year

# Other packages
import os
import matplotlib.pyplot as plt
import requests
import json
import seaborn as sns
import pprint
import re


%matplotlib inline
# Changing default plots
plt.style.use('default')  # set style (colors, background, size, gridlines etc.)
plt.rcParams['figure.figsize'] = 6, 4  # set default size of plots
plt.rcParams.update({'font.size': 10})
sns.set(style='ticks', palette="RdBu")
# for item in ax.get_yticklabels()+ax.get_xticklabels():
#     item.set_fontsize(12)

# Data
tips = sns.load_dataset('tips')


## Groupby
split_var = 'sex'
apply_var = 'total_bill'
tips\
    .groupby(split_var)\
    [apply_var]\
    .mean()

##########################################
#   Exercise Set 7: Data structuring 3   #
##########################################
## 7.1 weather data, part 3
df_64 = weather_for_year(1864)
df_64.head(3)
station = df_64['station'] == 'ITE00100550'
df_ITE = df_64[station].copy()
df_ITE['value'].describe()
df_ITE.head(3)
# Plot by month
split_var = ['month']
apply_var = 'value'
df_monthly = df_ITE\
    .groupby(split_var)\
    [apply_var]\
    .describe()


df_ITE['value'].plot(x='date', title='Max temperature').set_ylabel('Degrees celsius')






plt.legend()

# # Mathias
# split_var = ['station','month']
# apply_var = 'obs_value'
# monthly_weather = weather_1864\
#     .groupby(split_var)\
#     [apply_var]\
#     .describe()
#
# monthly_weather.loc['ITE00100550'].plot(y = ['mean','std','25%','75%','min','max'])







    # df_w['value'].plot(x='date', title='Max temperature').set_ylabel('Degrees celsius')
