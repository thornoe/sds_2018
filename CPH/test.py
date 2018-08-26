# import packages
import numpy as np
import pandas as pd
import time
# GeoPy - see https                            ://pypi.org/project/geopy/
import geopy.geocoders
from geopy.geocoders import Nominatim # retrieve coordinates from addresses etc.
geopy.geocoders.options.default_user_agent = 'my_app/1'
geopy.geocoders.options.default_timeout = 7

# read data
raw_data = pd.read_csv('CPH/Data/raw_data.csv')

# Keep the columns we want to keep and name them
sorted_data = raw_data.iloc[:,[0,2,3,4,7,8,12,14,15,16,18,21]]
sorted_data.columns = ['Address', 'Rooms', 'Area','Land_area','Owner_expense',
                       'Energy_mark', 'Price','Days_on_market',
                       'Zip_code', 'Town', 'Price_development', 'Sqm_price']

# Unpack all latitudes and longitudes and add to dataframe
longitude = []
latitude = []
raw_data['geometry']

for i, row in raw_data['geometry'].iteritems():
    if type(row) == str:
        coordinates = row.split('[', 1)[1]  # after '['
        longitude.append(float(coordinates.split(',', 1)[0]))  # before ','
        lat = coordinates.split(', ', 1)[1]  # after ', '
        latitude.append(float(lat.split(']', 1)[0]))  # before ']'
    else:
        longitude.append(None)
        latitude.append(None)
# Add latitude and longitude to sorted dataframe.
sorted_data.insert(loc=0, column='Latitude', value=latitude)
sorted_data.insert(loc=0, column='Longitude', value=longitude)

# Convert zip_code from str to int
sorted_data['Zip_code'] = sorted_data['Zip_code'].astype(int, inplace=True)
# Sort only zipcodes from Copenhagen
cph = sorted_data[(sorted_data.Zip_code < 3000)].copy()
cph.reset_index(inplace=True, drop=True)

# Make a column of indexes for energy mark
energysaving = []
for i, row in cph['Energy_mark'].iteritems():
    if row == 'G':
        energysaving.append(0)
    elif row == 'F':
        energysaving.append(1)
    elif row == 'E':
        energysaving.append(2)
    elif row == 'D':
        energysaving.append(3)
    elif row == 'C':
        energysaving.append(4)
    elif row == 'B':
        energysaving.append(5)
    elif row == 'A':
        energysaving.append(6)
    elif row == 'A10':
        energysaving.append(7)
    elif row == 'A15':
        energysaving.append(8)
    elif row == 'A20':
        energysaving.append(9)
    else:
        energysaving.append(None)
cph.insert(loc=0, column='Energy_saving', value=energysaving)

floor = []
for i, row in cph['Address'].iteritems():
    if ',' in row:
        sec_part = row.split(', ', 1)[1] # split once, keep 2nd part
        if sec_part[:2].isdigit():
            floor.append(None) # 399 with two or more digits (unindentified floor, 362 >= '20')
        elif sec_part[0].isdigit():
            floor_int = int(sec_part[0])
            floor.append(floor_int)
        else:
            floor.append(int(0))
    else:
        floor.append(int(0))
cph.insert(loc=0, column='Floor', value=floor)
##############################################################################
# Code latitude and longitude using GeoPy #
##############################################################################
address_simple = []
for i, row in cph['Address'].iteritems():
    if 'HUSBÅD' in row:
        address_simple.append(row.split(' - HUSBÅD', 1)[0])  # Keep first part
    elif 'Amerika Plads' in row:
        address_simple.append(row.replace('Plads', 'Pl.'))
    else:
        address_simple.append(row.split(',', 1)[0]) # split once, keep 1st part
cph.insert(loc=0, column='Address_simple', value=address_simple)

town_simple = []
for i, row in cph['Town'].iteritems():
    if 'København' in row:
        town_simple.append(row.split(' ', 1)[0])  # Keep 'København' only
    # elif "Frederiksberg C" in row:
    #     town_simple.append(row.split(' ', 1)[0])  # Keep 'Frederiksberg' only
    else:
        town_simple.append(row)
cph.insert(loc=0, column='Town_simple', value=town_simple)

cph['Full_address'] = cph['Address_simple'].map(str) + ', ' + cph['Zip_code'].map(str) + ' ' + cph['Town_simple'] # + ', Denmark'

# Thoose with missing latitude and longitude from scrape
cph['Missing'] = cph['Longitude'].isnull()
cph['Full_add'] = cph['Full_address']*cph['Missing']

# Retrieve coordinates from column of addresses
geolocator = Nominatim()
# geolocator.headers  # check header
# geolocator.timeout  # check time_out
lati = []
longi = []
# add = []

for row in cph['Full_add']:
    row_string = str(row)
    if len(row_string) > 0:
        location = geolocator.geocode(row_string)
        if isinstance(location, geopy.location.Location):
            lati.append(location.latitude)
            longi.append(location.longitude)
            # add.append(location.address)
        else:
            print('Not found: ',row_string)
            lati.append(None)
            longi.append(None)
    else:
        lati.append(None)
        longi.append(None)
    time.sleep(7)
print('Done!')

cph.insert(loc=0, column='Latitude', value=latitude)
cph.insert(loc=0, column='Longitude', value=longitude)
    # Create 'Floor' ---> make NANs into something weird
    # # Now sort columns in order which makes sence
# bolighed =cph.reindex(columns=['Address','Zip_code','Town', 'Latitude', 'Longitude' ,'Rooms',
# 'Area', 'Land_area','Sqm_price','Price','Owner_expense', 'Price_development',
# 'Energy_mark','Energy_saving', 'Days_on_market'])
return cph

# Search for a single item
location = geolocator.geocode('Århus Privatskole')
type(location)
location.address
isinstance(location, geopy.location.Location)
location.latitude
