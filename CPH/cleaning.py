
    import numpy as np
    import pandas as pd
    # GeoPy - see https                            ://pypi.org/project/geopy/
    import geopy.geocoders
    from geopy.geocoders import Nominatim # retrieve coordinates from addresses etc.
    geopy.geocoders.options.default_user_agent = 'my_app/1'
    geopy.geocoders.options.default_timeout = 5
    # geolocator.headers
    # geolocator.timeout
    raw_data = pd.DataFrame(data)
    # Keep the columns we want to keep and name them
    sorted_data = raw_data.iloc[:, [0,2,3,4,7,8,12,14,15,16,18,21]]
    sorted_data.columns = ['Address', 'Rooms', 'Area', 'Land_area', 'Owner_expense',
                           'Energy_mark', 'Price', 'Days_on_market',
                           'Zip_code', 'Town', 'Price_development', 'Sqm_price']

    # Convert zip_code from str to int
    sorted_data['Zip_code'] = sorted_data.Zip_code.astype(int, inplace=True)
    # Sort only zipcodes from Copenhagen
    cph = sorted_data[(sorted_data.Zip_code < 3000)].copy()
    cph.reset_index(inplace=True, drop=True)

    #Unpack all latitudes and longitudes and add to dataframe
    latitude = []
    for i in range(len(raw_data)):
        if raw_data['geometry'][i]!=None:
            row_i = raw_data['geometry'][i]['coordinates'][0]
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
    energysaving = []
    for keys, row in cph['Energy_mark'].iteritems():
        if row == 'G'                              :
            energysaving.append(0)
        elif row == 'F'                            :
            energysaving.append(1)
        elif row == 'E'                            :
            energysaving.append(2)
        elif row == 'D'                            :
            energysaving.append(3)
        elif row == 'C'                            :
            energysaving.append(4)
        elif row == 'B'                            :
            energysaving.append(5)
        elif row == 'A'                            :
            energysaving.append(6)
        elif row == 'A10'                          :
            energysaving.append(7)
        elif row == 'A15'                          :
            energysaving.append(8)
        elif row == 'A20'                          :
            energysaving.append(9)
        else                                       :
            energysaving.append(None)
    cph.insert(loc=0, column='Energy_saving', value=energysaving)

    ##############################################################################
    # Code latitude and longitude using GeoPy #
    ##############################################################################
    address_simple = []
    for keys, row in cph['Address'].iteritems()    :
        address_simple.append(row.split(',', 1)[0]) # split once, keep 1st part
    cph.insert(loc=0, column='Address_simple', value=address_simple)

    cph['Full_address'] = cph['Address_simple'].map(str) + ', ' + cph['Zip_code'].map(str) + ' ' + cph['Town']

    # Retrieve coordinates from column of addresses
    geolocator = Nominatim()
    latitude = []
    longitude = []
    # address = []

    for row in cph['Full_address']                 :
        row_string = str(row)
        location = geolocator.geocode(row_string)
        latitude.append(location.latitude)
        longitude.append(location.longitude)
        # address.append(location.address)
    cph.insert(loc=0, column='Latitude', value=latitude)
    cph.insert(loc=0, column='Longitude', value=longitude)

    # Create 'Floor'
    floor = []
    for keys, row in cph['Address'].iteritems()    :
        if ',' in row                              :
            sec_part = row.split(', ', 1)[1] # split once, keep 2nd part
            if sec_part[                           :2].isdigit():
                floor.append(None) # 399 with two or more digits (unindentified floor, 362 >= '20')
            elif sec_part[0].isdigit()             :
                floor_int = int(sec_part[0])
                floor.append(floor_int)
            else                                   :
                floor.append(int(0))
        else                                       :
            floor.append(int(0))
    cph.insert(loc=0, column='Floor', value=floor)

    # # Now sort columns in order which makes sence
    # bolighed =cph.reindex(columns=['Address','Zip_code','Town', 'Latitude', 'Longitude' ,'Rooms',
    # 'Area', 'Land_area','Sqm_price','Price','Owner_expense', 'Price_development',
    # 'Energy_mark','Energy_saving', 'Days_on_market'])
    return cph

# Search for a single item
# location = geolocator.geocode('Århus Friskole')
# location.address
# location.latitude
