# [Install packages]
!pip3 install tqdm
!pip3 install geopy
!conda install geopandas

# [Import packages]
import numpy as np
import pandas as pd

# Importing self-made functions
from CPH.scraping import scrape_bolighed
from CPH.cleaning import datastructuring

##############################################################################
#                    Scraping the data from bolighed.dk                      #
##############################################################################
# OBS ONLY RUN THIS ONCE. It takes around 18 minuttes.
# !pip3 install tqdm
raw_data = scrape_bolighed(5.0)  # ONLY RUN ONCE. Set sleep time to at least 5 seconds
raw_data = pd.DataFrame(raw_data)
raw_data.to_csv('CPH/Data/raw_data.csv', index=False)  # Save scraped data

##############################################################################
#                               Data cleaning                                #
##############################################################################
raw_data = pd.read_csv('CPH/Data/raw_data.csv')
cph = datastructuring(raw_data, 15)  # set default_timeout for GeoPy
cph.to_csv('CPH/Data/cph.csv', index=False)  # Save scraped data

##############################################################################
#                          Descriptive statistics                            #
##############################################################################
cph = pd.read_csv('CPH/Data/cph.csv')

# Take subset of table for descriptive statistics.
vars_ = ['Price', 'Owner_expense', 'Yearly_expenses', 'Sqm_price' , 'Sqm_owner_expenses', 'Sqm_yearly_expenses']
data_0 = cph.loc[:, vars_]
data_1 = data_0.describe()
# Round of decimal points and choose descriptives of interest.
cph_desc = data_1.astype(int)
cph_desc = cph_desc.iloc[[1,2,3,5,7],:]
descriptives = cph_desc.transpose()
print(descriptives)
descriptives.to_latex()  # Få tabel i LaTeX format:

# t-values of sqm_measures:
# 12069/43466  # sqm_price
# 9/41  # sqm_owner_expenses
# 429/1969  # sqm_yearly_expenses
##############################################################################
#              Implementing the K-means Clustering algorithm                 #
##############################################################################
cph = pd.read_csv('CPH/Data/cph.csv')
cph.isnull().sum()

##############################################################################
#                            Machine learning                                #
##############################################################################



##############################################################################
#                            Dropping extremes                               #
##############################################################################
data_apart = pd.read_csv('CPH/Data/data_apar.csv')
data_no_extremes = data_apart.sort_values(by=['Sqm_price']).copy()
data_no_extremes = data_no_extremes.iloc[25:-25]
data_no_extremes.sort_index(inplace=True)
