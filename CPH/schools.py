# [Getting latitude and longitude for schools]
# Import packages
!pip3 install tqdm
!pip3 install geopy
import numpy as np
import pandas as pd
import time, tqdm
import geopy.geocoders  # GeoPy - see https://pypi.org/project/geopy/
from geopy.geocoders import Nominatim # retrieve coordinates from addresses etc.
geopy.geocoders.options.default_user_agent = 'my_app/1'
geopy.geocoders.options.default_timeout = 1

# read data
schools = pd.read_csv('')  # UNFILLED

geolocator = Nominatim()
# geolocator.headers  # check header
# geolocator.timeout  # check time_out
latitude = []
longitude = []
address = []

for row in tqdm.tqdm(schools['Name']):
    row_string = str(row)
    location = geolocator.geocode(row_string)
    if isinstance(location, geopy.location.Location):
        latitude.append(float(location.latitude))
        longitude.append(float(location.longitude))
        address.append(float(location.address))
        time.sleep(5)
    else:
        print('Not found: ',row_string)
        latitude.append(None)
        longitude.append(None)
        time.sleep(5)
schools.insert(loc=0, column='Latitude', value=latitude)
schools.insert(loc=0, column='Longitude', value=longitude)
schools.insert(loc=0, column='Address', value=address)

# [Search for a single item in GeoPy:]
# import packages
# !pip3 install tqdm
# !pip3 install geopy
# import numpy as np
# import pandas as pd
# import time, tqdm
# import geopy.geocoders  # GeoPy - see https://pypi.org/project/geopy/
# from geopy.geocoders import Nominatim  # retrieve coordinates from addresses etc.
# geopy.geocoders.options.default_user_agent = 'my_app/1'
#
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
