# Import packages and set working directory
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import json
import seaborn as sns
import pprint

# Ex. 3.1.1 substring
s1 = 'Chameleon'
s2 = 'ham'
s2 in s1
'hello' in 'goodbye'

# Ex. 3.1.2 slicing
s1[-4:]
s1.find('a')

# Ex. 3.1.3 string formatting
l1 = ['r ', 'Is', '>', ' < ', 'g ', '?']
l1[0]='Is'
l1[1]='r'
l1.pop(3)  # remove item 4 and report it
l1 = [x.strip() for x in l1]  # remove spaces
l1 = ",".join(l1)
l1 = l1.replace(',',' ')
l1 = l1[:-2] + l1[-1]
l1

# Ex. 3.1.4 dictionaries
words={}
keys=['animal', 'coffee', 'python', 'unit',
    'knowledge', 'tread', 'arise']
values=[]

for letter in keys:
    if letter[0] in 'aeiou':
        values.append(bool(True))
    else:
        values.append(bool(False))
pairs = zip(keys,values)
words=dict(pairs)
words

# Ex. 3.1.5 .items()
for keys, values in words.items():
    if values==True:
        print(keys+' starts with a vowel')
    else:
        print(keys+' does not start with a vowel')

# Ex. 3.2.1
server_url = 'https://api.punkapi.com/v2/beers'
endpoint_path = '?brewed_before=12-2008&abv_gt=8'
url = server_url + endpoint_path
url

resp = requests.get(url)
resp.ok
len(resp.text)
resp.text[:500]
resp_json = resp.json()
resp_json



with open('my_file.json', 'w') as f:  # w: for writing, truncating first
    resp_json_str = json.dumps(resp.json())
    f.write(resp_json_str)


resp_json_str




pprint.pprint(response.json)


>> 1) How many beers are in the JSON object?
>>
>> 2) Print the names of the beers in the JSON object using lower case characters.
>>
>> 3) Select the beer called Paradox Islay from the JSON object.
>>
>> 4) Which hop ingredients does the Paradox Islay contain?
