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
df_64 = pd.read_csv("Exercises/weather1864.csv")
df_64.head(3)
df_station = df_64['identifier']=='ITE00100550'
df_ITE = df_64[df_station].copy()
df_ITE.describe()

df_ITE.describe(include=[])









    # df_w['value'].plot(x='date', title='Max temperature').set_ylabel('Degrees celsius')
