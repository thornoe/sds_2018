def datastructuring(data, timeout):
    # import packages
    import numpy as np
    import pandas as pd
    import time, tqdm
    import geopy.geocoders  # GeoPy - see https://pypi.org/project/geopy/
    from geopy.geocoders import Nominatim # retrieve coordinates from addresses etc.
    geopy.geocoders.options.default_user_agent = 'my_app/1'
    geopy.geocoders.options.default_timeout = timeout

    # read data
    raw_data = pd.DataFrame(data)

    # Keep the columns we want to keep and name them
    sorted_data = raw_data.iloc[:,[0,2,3,4,7,8,12,14,15,16,18,21]]
    sorted_data.columns = ['Address', 'Rooms', 'Area', 'Land_area', 'Owner_expense',
                           'Energy_mark', 'Price', 'Days_on_market',
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

    # Set all missing values with energy_saving to mean.
    cph.Energy_saving.fillna(5.0, inplace=True)

    # Create 'floor' variable
    floor = []
    for i, row in cph['Address'].iteritems():
        if ',' in row:
            sec_part = row.split(', ', 1)[1] # split once, keep 2nd part
            if sec_part[:2].isdigit():  # 399 with two or more digits (unindentified floor, 362 >= '20')
                floor_int = int(sec_part[0])  # assume 1st digit indicates floor
                floor.append(floor_int)
            elif sec_part[0].isdigit():
                floor_int = int(sec_part[0])
                floor.append(floor_int)
            else:
                floor.append(int(0))
        else:
            floor.append(int(0))
    cph.insert(loc=0, column='Floor', value=floor)

    ##########################################################################
    #           Code missing latitudes and longitudes using GeoPy            #
    ##########################################################################
    address_simple = []
    for i, row in cph['Address'].iteritems():
        if 'George Marshalls Vej' in row:
            address_simple.append('Fiskerihavnsgade 8')
        elif 'Amerika Plads' in row:
            address_simple.append(row.replace('Plads', 'Pl.'))
        elif 'HUSBÅD' in row:
            address_simple.append(row.split(' - HUSBÅD', 1)[0])  # Keep first part
        else:
            address_simple.append(row.split(',', 1)[0]) # split once, keep 1st part
    cph.insert(loc=0, column='Address_simple', value=address_simple)

    town_simple = []
    for i, row in cph['Town'].iteritems():
        if 'København' in row:
            town_simple.append('København')  # Keep 'København' only
        elif 'Nordhavn' in row:
            town_simple.append('København')  # Keep 'København' only
        else:
            town_simple.append(row)
    cph.insert(loc=0, column='Town_simple', value=town_simple)
    cph['Full_address'] = cph['Address_simple'].map(str) + ', ' + cph['Zip_code'].map(str) + ' ' + cph['Town_simple']

    # Thoose with missing latitude and longitude from scrape
    cph['Missing'] = cph['Longitude'].isnull()
    cph['Full_add'] = cph['Full_address']*cph['Missing']

    # Retrieve coordinates from column of addresses
    geolocator = Nominatim()
    # geolocator.headers  # check header
    # geolocator.timeout  # check time_out
    lati = []
    longi = []

    for row in tqdm.tqdm(cph['Full_add']):
        row_string = str(row)
        if len(row_string) > 0:
            location = geolocator.geocode(row_string)
            if isinstance(location, geopy.location.Location):
                lati.append(float(location.latitude))
                longi.append(float(location.longitude))
            else:
                print('Not found: ',row_string)
                lati.append(None)
                longi.append(None)
        else:
            lati.append(None)
            longi.append(None)
    cph.insert(loc=0, column='Lati', value=lati)
    cph.insert(loc=0, column='Longi', value=longi)
    cph['Latitude'] = cph['Latitude'].fillna(cph['Lati'])
    cph['Longitude'] = cph['Longitude'].fillna(cph['Longi'])
    cph.isnull().sum()
    # Drop columns
    # cph = cph.drop(['Longi', 'Lati', 'Town_simple', 'Address_simple',
    #     'Energy_mark', 'Town', 'Full_address', 'Missing', 'Full_add'], axis=1)

    ##########################################################################
    #                      Append municipality onto dataset                  #
    ##########################################################################
    # Get data from Statistics Denmark with zip_code:
    # Get zip codes and municipalities
    url_post = 'https://www.dst.dk/ext/4393839853/0/kundecenter/Tabel-Postnumre-kommuner-og-regioner--xlsx'
    df_muni = pd.read_excel(url_post)
    df2_muni = df_muni[4:]
    df2_muni.rename(columns={'Postnumre, kommuner og regioner, 1.1.2016':'Zip','Unnamed: 1':'Municipality','Unnamed: 2':'Region'}, inplace=True)

    # Split data: we want to seperate zip code and village as well as municipality number and municipality
    zip_split = pd.DataFrame(df2_muni.Zip.str.split(' ',1).tolist(),
                                       columns = ['Zip','Village'])

    mun_split = pd.DataFrame(df2_muni.Municipality.str.split(' ',1).tolist(),
                                       columns = ['Mun. no.','Municipality'])

    merge = pd.concat([zip_split, mun_split], axis=1, sort=False)

    #Construct new variable that only contain municpalities with zip code below 3000
    mun_zip = merge[['Zip','Municipality']]
    mun_zip['Int zip'] = mun_zip['Zip'].astype(int)
    our_sample = mun_zip[(mun_zip['Int zip'] < 3000)]
    # Drop string zip_code column:
    our_sample.drop('Zip', axis=1 ,inplace=True)
    # Give common name to this datasets zip code and our:
    our_sample.rename(columns={'Int zip': 'Zip_code'}, inplace=True)
    # Keep last duplicate of kommunes
    our_sample.drop_duplicates(keep='last', subset='Zip_code', inplace=True)
    # Now merge datasets on zip code:
    cph_merged= pd.merge(cph, our_sample, on='Zip_code', how='left')

    # Sort columns in order which makes sence
    cph_kom =cph_merged.reindex(columns=['Address', 'Zip_code', 'Municipality',
        'Latitude', 'Longitude', 'Floor', 'Rooms', 'Area', 'Land_area',
        'Sqm_price', 'Price', 'Owner_expense', 'Price_development',
        'Energy_saving', 'Days_on_market'])

    # Create log variable of square meter price
    cph_kom.insert(loc=9, column='log_sqm_price', value=np.log(cph_kom.Sqm_price))

    # Standard-Finance
    yearly_expenses = []
    first_year_expenses = []
    for houseprice, ownerexp in zip(cph_kom.Price, cph_kom.Owner_expense):
        """Vi antager at køberne selvfinansiere de 20 % af købssummen således:"""
        remaining = houseprice % 5000  # rest, når der deles med 5.000
        price = houseprice*0.05
        if remaining < 2500:
            price = int(price / 5000) * 5000  # runder ned
        else:
            price = int((price + 5000) / 5000) * 5000  # runder op
        Cashticket = max(price,25000)  # Kontant udbetaling på 5% af den kontante købesum oprundet til nærmeste kr. 5.000 dog minimum kr. 25.000.
        Mortgage = (houseprice*0.8)*0.03  # Der lånes 80% i realkreditinstitutet til en ÅOP på 3% efter skat.
        Bankloan = (houseprice*0.2 - Cashticket)*0.066  # De resterende ca. 15% lånes i banken til en ÅOP på 6.6% efter skat banklånet
        yearly_expenses.append(12*ownerexp + Bankloan + Mortgage)
        first_year_expenses.append(12*ownerexp + Bankloan + Mortgage + Cashticket)
    cph_kom.insert(loc=13, column='Yearly_expenses', value=yearly_expenses)
    cph_kom.insert(loc=14, column='First_year_expenses', value=first_year_expenses)
    log_yearly_sqm_exp = cph_kom.Yearly_expenses / cph_kom.Area
    cph_kom.insert(loc=15, column='log_yearly_sqm_exp', value=log_yearly_sqm_exp)

    print(cph_kom.isnull().sum())

    return cph_kom

# [Search for a single item in GeoPy:]
# import packages
# !pip3 install tqdm
# !pip3 install geopy
# import numpy as np
# import pandas as pd
# import time, tqdm
# import geopy.geocoders  # GeoPy - see https://pypi.org/project/geopy/
# from geopy.geocoders import Nominatim # retrieve coordinates from addresses etc.
# geopy.geocoders.options.default_user_agent = 'my_app/1'
# geolocator = Nominatim()
# location = geolocator.geocode('Sølvgade Skole')
#
# if isinstance(location, geopy.location.Location):
#     print('Address:', location.address,
#         '\nLatitude:', location.latitude,
#         '\nLongitude:', location.longitude)
# else:
#     print(type(location),
#     '\nWas found:', isinstance(location, geopy.location.Location))
