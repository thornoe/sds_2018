# [Import packages]
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests, json, tqdm, time
# import os, random, datetime, re, time, requests, json

%matplotlib inline



def datastructuring(data):
    raw_data=pd.DataFrame(data)
    #Vælg de kolonner vi vil beholde og giv kolonner navne
    sorted_data = raw_data.iloc[:,[0,2,3,4,7,8,12,14,15,16,18,21]]
    sorted_data.columns = ['Address', 'Rooms', 'Area','Land_area','Owner_expense',
                           'Energy_mark', 'Price','Days_on_market',
                           'Zip_code', 'Town', 'Price_development', 'Sqm_price']
    #Unpack all latitudes and longitudes and add to dataframe
    latitude = []
    for i in range(len(raw_data)):
        if raw_data['geometry'][i]!=None:
            row = raw_data['geometry'][i]['coordinates'][0]
            latitude.append(row_i)
        else:
            row_i=None
            latitude.append(row_i)
    longitude = []
    for i in range(len(raw_data)):
        if raw_data['geometry'][i]!=None:
            row_i = raw_data['geometry'][i]['coordinates'][1]
            longitude.append(row_i)
        else:
            row_i=None
            longitude.append(row_i)
    # Add latitude and longitude to sorted dataframe.
    sorted_data.insert(loc=0, column='Latitude', value=latitude)
    sorted_data.insert(loc=0, column='Longitude', value=longitude)
    # Make a column of indexes for energy mark
    Energysaving = []
    for keys, row in sorted_data['Energy_mark'].iteritems():
        if row == 'G':
            Energysaving.append(0)
        elif row == 'F':
             Energysaving.append(1)
        elif row == 'E':
             Energysaving.append(2)
        elif row == 'D':
             Energysaving.append(3)
        elif row == 'C':
             Energysaving.append(4)
        elif row == 'B':
             Energysaving.append(5)
        elif row == 'A':
             Energysaving.append(6)
        elif row == 'A10':
             Energysaving.append(7)
        elif row == 'A15':
             Energysaving.append(8)
        elif row == 'A20':
             Energysaving.append(9)
        else:
             Energysaving.append(None)
    sorted_data.insert(loc=0, column='Energy_saving', value=Energysaving)
    # Now sort columns in order which makes sence
    bolighed =sorted_data.reindex(columns=['Address','Zip_code','Town', 'Latitude', 'Longitude' ,'Rooms',
                                        'Area', 'Land_area','Sqm_price','Price','Owner_expense', 'Price_development',
                                        'Energy_mark','Energy_saving', 'Days_on_market'])
    # Convert zip_code from str to int
    bolighed['Zip_code']=bolighed.Zip_code.astype(int, inplace=True)
    # Sort only zipcodes from Copenhagen
    cph = bolighed[(bolighed.Zip_code < 3000)].copy()
    cph.reset_index(inplace=True, drop=True)
    print('Now we have a sorted data set with ' + str(len(cph)) + ' observations.')
    return cph

# [Run data cleaning code]
raw_data = pd.read_csv('CPH/Data/raw_data.csv')
cph = datastructuring(raw_data)
