# General packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pprint as p
import random
import datetime

## Packages for scraping
import requests, re, tqdm, time, json
from bs4 import BeautifulSoup  # for parsing
from selenium import webdriver

## plot styles
%matplotlib inline
sns.set_style('white')
plt.style.use('seaborn-white')

################################################################################
#   Self-defined function for efficient and transparent scraping               #
################################################################################
s = requests.session()
s.headers['name'] = 'Thor Donsby Noe (student)' # telling who I am
s.headers['email'] = 'jwz766@alumni.ku.dk' # allowing them to contact me
s.headers['User-Agent'] = 'Mozilla/5.0'
# 'User-Agent' is sometimes required.
# Tip: change the user agent to a cell phone to obtain more simple formatting of the html, e.g.
# s.headers['User-Agent'] = 'Mozilla/5.0 (iPhone; CPU iPhone OS 10_2_1 like Mac OS X) AppleWebKit/602.4.6 (KHTML, like Gecko) Version/10.0 Mobile/14D27 Safari/602.1'
s.headers # show headers

def get(url, iterations=20, sleep_ok=1, sleep_err=5, check_function=lambda x: x.ok):
    """This module ensures that your script does not crash from connection errors,
        that you limit the rate of your calls,
        and that you have some reliability check.
        iterations : Define number of iterations before giving up.
    """
    for iteration in range(iterations):
        try:
            time.sleep(sleep_ok) # sleep everytime
            response = s.get(url) # the s-function defined above
            if check_function(response):
                return response # if succesful it will end the iterations here
        except requests.exceptions.RequestException as e: # find exceptions in the request library requests.exceptions
            """ Exceptions: Define which exceptions you accept, default is all.
            For specific errors see:
            stackoverflow.com/questions/16511337/correct-way-to-try-except-using-python-requests-module
            """
            print(e)  # print or log the exception message
            time.sleep(sleep_err) # sleep before trying again in case of error
    return None # code will purposely crash if you don't create a check function later

################################################################################
#    Automatic browsing - programming your browser to do stuff on it's own     #
################################################################################
"""
for automatic browsing / screen scraping (not covered in detail here): 'selenium'
for behaving responsibly we use: 'time' and 'our minds'
We will write our scrapers with basic python,
   for larger projects consider looking into the packages scrapy or pyspider

Adress: www.google.com
Instructions: /trends?query=social+data+science
Header: information send along with the request, including user agent
   (operating system, browser), cookies, and prefered encoding.
   set to 'smartphone'
"""

################################################################################
#    Collecting data on display - lets you choose between tables directly      #
################################################################################
## Basic examples
url = 'https://www.basketball-reference.com/leagues/NBA_2018.html' # link to the website
dfs = pd.read_html(url)  # parses all tables found on the page.
len(dfs)  # number of tables
dfs[0]  # showing 1st table

# response.text.split("   ")[1]  # discarding the 1st part (within parenthesis)

## search for specific tables
EC_df = pd.read_html(url,attrs={'id':'confs_standings_E'})  # only parse the tables with attribute confs_standings_E

################################################################################
#   Collecting data BEHIND the display                                         #
################################################################################
"""
Open (or install) Firefox, right-click on a graphic to "Inspect element".
Go to "Network" banner, choose XHR, update (F5) and look for JSON files.
- Go through the JSON files that are not 3rd part (advertisement).
- In the panel in the bottom right first look under "Response" for the data
- When the desired JSON-file is found, go to "Headers" for the "Request-URL"
"""

url = 'https://graphs2.coinmarketcap.com/currencies/bitcoin/'
# response = requests.get(url)
response = get(url)  # self-defined function
response.ok
response.text
d = requests.get(url).json()
d.keys()
d['price_usd']
x,y = zip(*d['price_usd'])
### Error: shows seconds on January 1 in 1970
# t = [pd.to_datetime(t) for t in x]
# t = [datetime.datetime.fromtimestamp(int(i)) for i in x]
# plt.plot(t,y,label='usd')  # you can add as many as you want of theese!
# plt.yscale('log')
# plt.title('Bitcoin price in USD')
# plt.ylabel('log USD')
# plt.show()

################################################################################
#   Collecting unstructured data from kurser.ku.dk                             #
################################################################################
"""
Hint for crating a list via a loop:
    study_years = [str(y)+"-"+str(y+1) for y in range(first_year, last_year)]
is just a shorter version of writing:
    study_years = []
    for y in range(first_year, last_year):
        s = str(y)+"-"+str(y+1)
        study_years.append(s)
"""

def study_boards(first_year, last_year):
    """ Input: The first and last year considered
        e.g. 2013 and 2014 to get the single study year 2013-2014.
        Output: The links for all of the study boards.
    """
    study_years = [str(y)+"-"+str(y+1) for y in range(first_year, last_year)]
    all_years = []
    for study_year in tqdm.tqdm(study_years):
        url = 'https://kurser.ku.dk/archive/'+study_year
        response = get(url)
        html = response.text
        study_board_locations = html.split('href="')[1:]
        all_links = [study_board_loc.split('"')[0] for study_board_loc in study_board_locations]
        study_boards = [study_board for study_board in all_links if str(study_year) in study_board]
        study_board_links = ['https://kurser.ku.dk/archive/'+study_board for study_board in study_boards]
        all_study_boards = []
        for study_board_link in tqdm.tqdm(study_board_links):
            url = str(study_board_link)
            response = get(url)
            html = response.text
            link_locations = html.split('href="')[1:]
            all_links = [link_loc.split('"')[0] for link_loc in link_locations]
            links = ['https://www.kurser.ku.dk'+link for link in all_links if '/archive/' in link]
            all_study_boards += links # += joins two lists
        all_years += all_study_boards
    return all_years

links = study_boards(first_year = 2015, last_year = 2017)

len(links)
random.sample(links,10)

################################################################################
#   How URLS are constructed                                                   #
################################################################################
"""
/ is like folders on your computer.
? entails the start of a query with parameters
= defines a variable: e.g. page=1000 or offset=100 or showNumber=20
& separates different parameters.
+ is html for whitespace
"""

################################################################################
#   Exercise 8                                                                 #
################################################################################
### 8.1.2: Collect the 1st page
url = 'https://job.jobnet.dk/CV/FindWork/Search'
resp = get(url) # We use the request module to collect the first page of job postings
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
