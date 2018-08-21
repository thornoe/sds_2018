import warnings
import numpy as np
import pandas as pd

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
    df_w = df_w.set_index('date')  # to keep date-variable: df_w.set_index(df_time, inplace=True)
    df_w['country'] = df_w['station'].str[0:3]
    print('DataFrame created as follows')
    print(df_w.head(3))
    return df_w

# # run by defining a dataframe and applying a year
# df_w = weather_for_year(1864)
# df_w.head(3)

## Import using
# from Exercises.weather import weather_for_year

## Save using relative path
# df_w.to_csv('Exercises/weather1864.csv', index=False)  # default is to save the index
