import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pprint as p
import requests, random, datetime, re, time

## Packages for scraping
from bs4 import BeautifulSoup  # for parsing
from selenium import webdriver
import explore_regex as e_re
import json

%matplotlib inline
## Functions
pp = p.PrettyPrinter()  # pp.pprint(

## plot styles
sns.set_style('white')
plt.style.use('seaborn-white')

###############################################################
#              Interactions and Automated Browsing            #
###############################################################
# Sometimes scraping tasks demand interactions (e.g. login, scrolling, clicking),
#  and a no XHR data can be found easily, so you need the browser to execute
#  the scripts before you can get the data.
# Dowlnoad the latest release, and installing the latest selenium version.
# pip3 install selenium --upgrade
path2gecko = "D:\Opsætning\geckodriver.exe"  # define path to your geckodriver
browser = webdriver.Firefox(executable_path=path2gecko) # start the browser with a path to the geckodriver.
browser.get('https://www.google.com') # opens a webpage using the browser objects get method.
# Selenium will not find links within an iframe unless you search 'iframe'

###############################################################
#                Parsing HTML with BeautifulSoup              #
###############################################################
### scraping newspaper articles example.
url = 'https://www.theguardian.com/us-news/2018/aug/10/omarosa-trump-book-the-apprentice-memoir'
response = requests.get(url)
html = response.text
browser.get(url)

soup = BeautifulSoup(html,'lxml')  # parse the raw html using BeautifoulSoup

# extract hyperlinks
links = soup.find_all('a')  # find all a tags -connoting a hyperlink.
[link['href'] for link in links if link.has_attr('href')][-5:]  # unpack the hyperlink from the a nodes.

headline = soup.find('h1')  # search for the first headline: h1 tag.
headline.text.replace('Trump', 'the idiot')

name = headline['class'][0].strip()  # use the class attribute name as column name.
value = headline.text.strip()  # extract text using build in method.
print({name:value})
## Inspect object in Firefox
article_text = soup.find('div',{'class':'content__article-body from-content-api js-article__body'}).text  # find the content.

# Find links WITHIN the body of the text using BeautifulSoup
# find the article_content node
article_content = soup.find('div',{'class':'content__article-body from-content-api js-article__body'})

## find citations within the article content.
citations = article_content.find_all('a')

citation_links = []  # define container to the hyperlinks
for citation in citations:  # iterate through each citation node
    if citation.has_attr('data-link-name'):  # check if it has the right attribute
        if citation['data-link-name'] == 'in body link':  # and if the value of that attribute is correct
            # print(citation['href'])
            citation_links.append(citation['href'])  # add link to the container

###############################################################
#    Extracting patterns from Raw Text: Regular Expressions   #
###############################################################
### Basic string operationg:
# string.split
# string.strip
# string.replace

### Regular Expressions: Special charecters
# https://www.regular-expressions.info/refquick.html
# + = 1 or more times -- e.g. "a+" will match: "a", and "aaa"
# * = 0 or more times -- e.g. "ba*" will match: "b", and "ba", and "baaa"
# {3} = exactly three times --- e.g. "ba{3}" will match "baaa", but not "baa"
# ? = once or none
# \ = escape character, used to find characters that has special meaning with regex: e.g. + *
# [] = allows you to define a set of characters
# ^ = applied within a set, it becomes the inverse of the set defined. Applied outside a set it entails the beginning of a string. $ entails the end of a string.
# . = any characters except line break
# | = or statement. -- e.g. a|b means find characters a or b.
# \d = digits
# \D = any-non-digits.
# \s = whitespace-separator
## Sequences:
# (?:) = Defines a Non-capturing group. -- e.g. "(?:abc)+", will match "abc" and "abcabcabc", but not "aabbcc"
# (?=) = Positive lookahead - only match a certain pattern if a certain pattern comes after it.
# (?!) = Negative lookahead - only match a certain pattern if not a certain pattern comes after it.
# (?<=) = Positive lookbehind - only match a certain pattern if a certain pattern precedes it.
# (?<!) = Negative lookbehind - only match a certain pattern if not a certain pattern precedes it.

###############################################################
#      Regular expressions (2): define - inspect - refine     #
###############################################################
# download module
url = 'https://raw.githubusercontent.com/snorreralund/explore_regex/master/explore_regex.py'
response = requests.get(url)
# write script to your folder to create a locate module
with open('explore_regex.py','w') as f:
    f.write(response.text)
# import local module
import explore_regex as e_re

### Example
path2data = 'https://raw.githubusercontent.com/snorreralund/scraping_seminar/master/danish_review_sample.csv'
df = pd.read_csv(path2data)
df.to_csv('danish_review_sample.csv',index=False)

digit_re = re.compile('[0-9]+') # compiled regular expression for matching digits
df['hasNumber'] = df.reviewBody.apply(lambda x: len(digit_re.findall(x))>0) # check if it has a number

sample_string = '\n'.join(df[df.hasNumber].sample(2000).reviewBody)

### money example
# explore_money = ExploreRegex(sample_string)
explore_money = e_re.ExploreRegex(sample_string)
pattern = '[0-9]+(?:[.,][0-9]+) ?kr'  # ?: special character for zero/none (space doesn't have to be there)
explore_money.explore_pattern(pattern)
explore_money.report('kr')

# first = 'kr'
# second = '[0-9]+kr'
# third = '[0-9]+(?:[,.][0-9]+)?kr'
# fourth = '[0-9]+(?:[,.][0-9]+)?\s{0,2}kr'
# final = '[0-9]+(?:[,.][0-9]+)?\s{0,5}kr(?:oner)?'
# patterns = [first,second,third,fourth,final]
# for pattern in patterns:
#     explore_money.explore_difference(pattern,patterns[0])
# explore_money.explore_pattern(second)



#########################################################################
#   Exercise Sectino 9.1: Parsing a Table from HTML w. BeautifulSoup    #
#########################################################################
### 9.1.1 Find table node
url = 'https://www.basketball-reference.com/leagues/NBA_2018.html'
browser.get(url)
response = requests.get(url)
html = response.text
soup = BeautifulSoup(html, 'lxml')

table = soup.find('table')

### 9.1.2 Create first column
# parse the header which can be found in the canonical tag name: thead
# use the `find_all` method to search for the tag
# iterate through each of the elements extracting the text,
# using the `.text` method builtin to the the node object
# Store the header values in a list container.
header = table.find('thead')
columns = []
for column in header.find_all('th'):
    columns.append(column.text)

columns

# Create a df with the columns

### 9.1.3 Locate the rows
tbody = table.find('tbody')

df = []
### 9.1.4 iterate through each row
rows = tbody.find_all('tr')
for i in range(len(rows)):
    row_i = rows[i]
    list_i = []
    for val_node in row_i.children:
        list_i.append(val_node.text)
    df.append(list_i)
df = pd.DataFrame(df, columns=columns)
df.set_index('Eastern Conference', inplace=True)

pp.pprint(row)

#########################################################################
#            Exercise Section 9.2: Text search in reviews               #
#########################################################################
### 9.2.1
# Table of entire data
df = pd.read_csv('https://raw.githubusercontent.com/snorreralund/scraping_seminar/master/english_review_sample.csv')
# List of all reviews
sample_string = '\n'.join(df.sample(2000).reviewBody)
# sample_string = '\n'.join(df[df.hasNumber].sample(2000).reviewBody)
# print(sample_string)

### 9.2.2
explore_money = e_re.ExploreRegex(sample_string)
d1 = '\d*\.*\d*\$\d*\.*\d*'
d2 =  '\$\s?\d+\.*\d*'
d3 = '[\d+\.\d*][\s][\$]'
d4 = '[\$][\s]\d*\.\d*.'
d5 = '\d+\s{1,}dollar.'
d6 = '\d+\s?[USD|usd].'

patterns = [d1, d2, d3, d4, d5, d6]

for pattern in patterns:
    explore_money.explore_difference(pattern, patterns[0])

explore_money.explore_pattern(d5)  # explore context

### 9.2.3 print all patterns
explore_money.report()
