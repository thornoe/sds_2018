import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pprint as p
import random
import datetime

## Packages for scraping
import requests, re, time, json  # for connecting to the internet
from bs4 import BeautifulSoup  # for parsing
from selenium import webdriver
import explore_regex as e_re
def get(url, iterations=10, check_function=lambda x: x.ok):
    """This module ensures that your script does not crash from connection errors.
        that you limit the rate of your calls
        that you have some reliability check
        iterations : Define number of iterations before giving up.
        exceptions: Define which exceptions you accept, default is all.
    """
    for iteration in range(iterations):
        try:
            "A function that handles the rate of your calls."
            time.sleep(1) # sleep one second.
            response = session.get(url)
            if check_function(response):
                return response # if succesful it will end the iterations here
        except exceptions as e: #  find exceptions in the request library requests.exceptions
            print(e) # print or log the exception message.
    return None # your code will purposely crash if you don't create a check function later.

## Functions
pp = p.PrettyPrinter()  # pp.pprint()

%matplotlib inline
## plot styles
sns.set_style('white')
plt.style.use('seaborn-white')

# for automatic browsing / screen scraping (not covered in detail here): 'selenium'
# for behaving responsibly we use: 'time' and 'our minds'
# We will write our scrapers with basic python,
#    for larger projects consider looking into the packages scrapy or pyspider

# Adress: www.google.com
# Instructions: /trends?query=social+data+science
# Header: information send along with the request, including user agent
#    (operating system, browser), cookies, and prefered encoding.
#    set to 'smartphone'

###############################################################################
#    Collecting data on display - lets you choose between tables directly     #
###############################################################################
## Basic examples
url = 'https://www.basketball-reference.com/leagues/NBA_2018.html'  # link to the website
dfs = pd.read_html(url)  # parses all tables found on the page.
len(dfs)  # number of tables
dfs[0]  # showing 1st table

# response.text.split("   ")[1]  # discarding the 1st part (within parenthesis)

## search for specific tables
EC_df = pd.read_html(url,attrs={'id':'confs_standings_E'})  # only parse the tables with attribute confs_standings_E

#############################################
#   Collecting datas BEHIND the display     #
#############################################
# right-click to "Inspect element"
url = 'https://graphs2.coinmarketcap.com/currencies/bitcoin/'
# response = requests.get(url)
resonse = get(url)  # self defined function
response.ok
response.text
d = requests.get(url).json()
d.keys()
d['price_usd']
x,y = zip(*d['price_usd'])
### Error: shows seconds on January 1 in 1970
# t = [pd.to_datetime(t) for t in x]
# t = [datetime.datetime.fromtimestamp(int(i)) for i in x]
#
# plt.plot(t,y,label='usd')  # you can add as many as you want of theese!
# plt.yscale('log')
# plt.title('Bitcoin price in USD')
# plt.ylabel('log USD')
# plt.show()

###############################################################
#      Collecting unstructured data from kurser.ku.dk         #
###############################################################
study_year = '2016-2017'
def study_boards(study_year):  # from 2013/2014 to 2016/2017


url = 'https://kurser.ku.dk/archive/'+str(study_year)
response = get(url)
# response.ok
html = response.text
study_board_locations = html.split('href="')[1:]
study_boards = []
for study_board_loc in study_board_locations:
    study_board = study_board_loc.split('"')[0]
    study_boards.append(study_board)
study_board = [study_board for study_board in study_boards if str(study_year) in study_board]
study_board_links = ['https://kurser.ku.dk/archive/'+study_board for study_board in study_board]

for study_board_link in study_board_links:
    url = str(study_board_link)
    response = get(url)
    html = response.text
    link_locations = html.split('href="')[1:]
    links = []
    for link_loc in link_locations:
        link = link_loc.split('"')[0]
        links.append(link)
    link = [link for link in links if '/archive/' in link]
    links = ['https://www.kurser.ku.dk'+link for link in link]

len(links)
random.sample(links,5)

###############################################################
#                 How URLS are constructed                    #
###############################################################
# / is like folders on your computer.
# ? entails the start of a query with parameters
# = defines a variable: e.g. page=1000 or offset = 100 or showNumber=20
# & separates different parameters.
# + is html for whitespace

###############################################################
#                   Transparent scraping                      #
###############################################################
response = requests.get('https://www.google.com')
session = requests.session()
session.headers['email'] = 'jwz766@alumni.ku.dk'
session.headers['name'] = 'Thor Donsby Noe'

# session.headers['User-Agent'] = '' # sometimes you need to pose as another agent...
# e.g. A quick tip is that you can change the user agent to a cellphone
#  to obtain more simple formatting of the html.
session.headers

###############################################################
#                           Exercise 8                        #
###############################################################
### 8.1.2: Collect the 1st page
url = 'https://job.jobnet.dk/CV/FindWork/Search'
resp = requests.get(url)        # We use the request module to collect the first page of job postings
jobnet = json.loads(resp.text)
jobnet

# Unpack the json data into a pandas DataFrame
jobnet.keys()  # Use this command to get an idea of what the data contains
jobnet_1 = pd.DataFrame(jobnet['JobPositionPostings'])  # Creates a Panda DataFrame of job postings
jobnet_1.head()  # Prints head of DataFrame

### 8.1.3
# Create a variable with the number of job postings
antal_stillinger = jobnet['TotalResultCount']
antal_stillinger
