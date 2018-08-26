import pandas as pd
import requests, json
import plotly.offline as py # import plotly in offline mode
py.init_notebook_mode(connected=True) # initialize the offline mode, with access to the internet or not.
import plotly.tools as tls
tls.embed('https://plot.ly/~cufflinks/8') # embed cufflinks.
# import cufflinks and make it offline
import cufflinks as cf
cf.go_offline() # initialize cufflinks in offline mode

# find the data using request at https://raw.githubusercontent.com/snorreralund/sds_dump/tsne_name_embedding
# df = pd.read_csv('tsne_name_embedding.csv')

#########################################
#   Scraping names from ankestyrelsen   #
#########################################
import pandas as pd
# links found looking here: https://ast.dk/born-familie/navne/navnelister/godkendte-fornavne
urls = {'female': 'https://ast.dk/_namesdb/export/names?format=xls&gendermask=1',
        'male': 'https://ast.dk/_namesdb/export/names?format=xls&gendermask=2',
        'unisex': 'https://ast.dk/_namesdb/export/names?format=xls&gendermask=3'}
dfs = {}
for name, url in urls.items():
    df = pd.read_excel(url,header)
    dfs[name] = df

# scrape ethnicity:
# Link located by looking in the network monitor for activity on this page:
# https://ast.dk/born-familie/navne/navnelister/udenlandske-navne
import requests
url = 'https://ast.dk/_namesdb/namecultures?published=1&page=1&pagesize=270'
response = requests.get(url)

import requests
url = 'https://ast.dk/_namesdb/namecultures?published=1&page=1&pagesize=270'
response = requests.get(url)
d = response.json()
import tqdm
import pandas as pd
data = []
for num in tqdm.tqdm(range(1,d['pages']+1)):
    url = 'https://ast.dk/_namesdb/namecultures?published=1&page=%d&pagesize=270'%(num)
    response = requests.get(url)
    d = response.json()
    data+=d['names']
df_eth = pd.DataFrame(data)
print(len(df_eth))

df_eth.head()

name2eth_gender = {}
for name,culture,sex, sex2 in df_eth[['name','culture','female','male']].values:
    if name in name2eth_gender:

        name2eth_gender[name]['culture'] = culture
        if sex&sex2:
            name2eth_gender[name]['sex'] = 'unisex'
            continue
        elif sex:
            name2eth_gender[name]['sex'] = 'female'
        elif sex2:
            name2eth_gender[name]['sex'] = 'male'
        else:
            print('eo')
    else:
        name2eth_gender[name] = {'culture':culture}
        if sex&sex2:
            name2eth_gender[name]['sex'] = 'unisex'
            continue
        elif sex:
            name2eth_gender[name]['sex'] = 'female'
        elif sex2:
            name2eth_gender[name]['sex'] = 'male'
        else:
            del name2eth_gender[name]
            #print(name,sex,sex2,'missing info')

names_used = set()
for sex,df in dfs.items():
    names = [df.columns[0]]+list(df.values[:,0])
    if sex=='unisex':
        continue
    print(sex,len(names),len(set(names)&set(name2eth_gender)))
    for name in names:
        if name in name2eth_gender:
            if name in names_used:
                name2eth_gender[name]['sex'] = 'unisex'
            continue
            name2eth_gender[name]['sex'].append(sex)
        else:
            name2eth_gender[name] = {'sex':sex}
            names_used.add(name)

from collections import Counter
Counter([i['culture'] for i in name2eth_gender.values() if 'culture' in i]),Counter([i['sex'] for i in name2eth_gender.values()])

json.dump(name2eth_gender,open('name2eth_gender_json','w'))

# download from 'https://raw.githubusercontent.com/snorreralund/sds_dump/name2eth_gender_json'

name2eth_gender = json.load(open('name2eth_gender_json','r'))
print(len(name2eth_gender))
import random
random.sample(list(name2eth_gender.items()),5)

#########################################
#            Lookup method              #
#########################################
# Lookups in Sets (and dictionaries) are really fast, and Lists are slow! So for lookups, do not use Lists.
female = set([name for name,d in name2eth_gender.items() if d['sex']=='female'])
male = set([name for name,d in name2eth_gender.items() if d['sex']=='male'])
male_l = list(male)  # convert set into a list
male_d = {name=1 for name in male}  # convert into a dictionary
print('Anders' in male)

%timeit 'Anders' in male
%timeit 'Anders' in male_l  # slow!
%timeit 'Anders' in male_d
# either do lookups in set containers or dictionary containers!

############################################
#   Inferring gender from review dataset   #
############################################
df = pd.read_csv('https://raw.githubusercontent.com/snorreralund/scraping_seminar/master/english_review_sample.csv')

def infer_gender(name):
    if name in name2eth_gender:
        return name2eth_gender[name]['sex']
    return 'unknown'
infer_gender('Anders')

print(infer_gender('Lars Ole'))
from collections import Counter
Counter([len(name.split()) for name in name2eth_gender])
# 216 names has more than one token, e.g. Lars Ole

# lookup scheme that take into account, that we need to ectract the first name from the
# full name, while also handling that some names are actually spanning more than one token
def infer_gender_ngram(full_name):
    """If more than one token in a name, looks up each name
    and the combination"""
    names = full_name.split()
    firstname = names[0]
    if firstname in name2eth_gender:
        return name2eth_gender[firstname]['sex']
    n_names = len(names)-1
    if n_names>=2:
        second_name = names[1]
        if second_name in name2eth_gender:
            return name2eth_gender[second_name]['sex']
        combinations = zip(*[names[i:] for i in range(n_names)])
        for comb in combinations:
            if comb in name2eth_gender:
                name = ' '.join(comb)
                return name2eth_gender[name]['sex']
    return 'unknown'
print(infer_gender_ngram('Coolio Snorre Ralund'))



#################################################
# Train the Word2Vec Model using Gensim.        #
#################################################
### Import modules ###
import gensim
import gensim.parsing.preprocessing
from gensim.parsing.preprocessing import preprocess_string

### APPLY Your Favorite Preprocessing + Tokenization + PostProcessing Scheme ###
filters = gensim.parsing.preprocessing.DEFAULT_FILTERS
filters = [i for i in filters[0:2]+filters[4:] if not i.__name__ =='strip_short']
corpus = [preprocess_string(text,filters) for text in df.text.values]
### Import model and logging function ###
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.models.word2vec import Word2Vec
### Define parameters for the model ###
size=300 # Size of Embedding.
workers=4 # Number CPU Cores to use for training in Parallel.
iter_=10 # Depending on the size of your data you might want to run through your data more than once.
window=6 # How much Context
min_count=5 # Number of Occurrences to be kept in the Vocabulary
### Initialize model and start training ###
model = Word2Vec(corpus,size=size,workers=workers,iter=iter_,window=window,min_count=min_count)

###########################################################################
# Big Data implementation: Streaming data instead of keeping it in memory #
###########################################################################
class DocumentStream(object):
    # Defines a class for streaming data from disk.
    # This specific one reads lines from a list of files, so this should adopted to your data(-base) setup.
    def __init__(self, files):
        #self.filename = filename
        self.files = files
    def __iter__(self):
        for filename in self.files:
            with codecs.open(filename,'r','utf-8') as f:
                for line in f:
                    yield preprocess_string(line,filters) # or apply your favorite preproccesing scheme
path2files = '/'
from os import listdir
files = [path2files+filename for filename in listdir(files)]
corpus = DocumentStream(files)
model = Word2Vec(corpus,size=size,workers=workers,iter=iter_,window=window,min_count=min_count)

#################################################
# Run latent dirichlet alloaction using Gensim. #
#################################################

### Import modules ###
import gensim
import gensim.parsing.preprocessing
from gensim.parsing.preprocessing import preprocess_string

### APPLY Your Favorite Preprocessing + Tokenization + PostProcessing Scheme ###
filters = gensim.parsing.preprocessing.DEFAULT_FILTERS
filters = [i for i in filters[0:2]+filters[4:] if not i.__name__ =='strip_short']
processed_docs = [preprocess_string(text,filters) for text in df.text.values]
### Create Index using gensims own method
dictionary = gensim.corpora.Dictionary(processed_docs)
### OPTIONAL: Filter Number of Dimensions ###
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
### Convert Tokenized Documents to BoWs using gensims own method ###
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
### Run model with K topics and define number of Cores for multiprocessing ###
n_cores = 8
k = 50
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=k, id2word=dictionary, passes=2, workers=n_cores)
### WAIT ###

#################################################
# Simple counting operations                    #
#################################################
import re,nltk
def count_pattern(text,pattern):
    return text.count(pattern)
def find_allcaps(text): # function that finds WORDS IN ALL CAPS
    pattern = r'[A-ZÆØÅ]{2,}'
    return len(re.findall(pattern,text))
def pseudo_lix(text): # Function that counts syllables and divides by number of words in a text
    words = nltk.word_tokenize(text)
    pattern = ur'[AEYUIOÅØÆaeyuioåæø]+'
    syllables = re.findall(pattern,text)
    return len(syllables)/len(words)
def count_regex_pattern(text,pattern,lower=True): # function allows you to count regex patterns
    if lower:
        text = text.lower()
    matches = re.findall(pattern,text)
    return len(matches)
sample['length'] = sample.message.apply(len)
sample['lix'] = sample.message.apply(lixtal)
sample['commas'] = [count_pattern(text,pattern=',') for text in sample.message]
sample['questions'] = [count_pattern(text,pattern='?') for text in sample.message]
sample['quotes'] = [count_pattern(text,pattern='"') for text in sample.message]
sample['caps'] = sample.message.apply(find_allcaps)
me_pattern = r'I|me'
we_pattern = r'we|us'
sample['me'] = [count_regex_pattern(text,me_pattern) for text in sample.message]
sample['we'] = [count_regex_pattern(text,we_pattern) for text in sample.message]
