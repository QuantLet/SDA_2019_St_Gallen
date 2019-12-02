"""
This file scrapes the news website finanzen.net and gathers specified articles,
analysis the words used and links them to the stocks return and volatility, following the publication of the article.

Stock data is retrieved through the Yahoo Finance API.

Code was written by Oliver Kostorz during November 2019.
"""

#Import packages
from bs4 import BeautifulSoup
import requests
import yfinance
from stop_words import get_stop_words
import re
import datetime
from fuzzywuzzy import process
import datetime as dt
import pandas
import numpy
import math
import pickle
import os
import json

#Set working device
try:
    wd = os.path.join(os.getcwd(), 'SDA-Oliver-Kostorz-SMART-Sentiment-Analysis-master')
except:
    print('Please specify path to working directory manually.')
    
###Functions

#Clean html tags in string
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)
    
#Get stock symbol
def getCompany(text):
    r = requests.get('https://api.iextrading.com/1.0/ref-data/symbols')
    stockList = r.json()
    return process.extractOne(text, stockList)[0]

#Rounds time downwards to earlier five minute interval
def roundtime(time):
    while 1<2:
        if float(time.minute/5).is_integer() is True:
            break
        time += dt.timedelta(minutes = -1)
    return time


###Data mining

#List containing relevant URLs for news-sample
with open(os.path.join(wd, 'links.txt'), "rb") as fp:   # Unpickling
    url_list = pickle.load(fp)

#Add additional links to news articles to use for training the algorithm
#Refer to Meta-Information for full guide and requirements
try:
    url_list.append()
except:
    pass

with open(os.path.join(wd, 'links.txt'), "wb") as fp:   #Pickling
    pickle.dump(url_list, fp)
    
#Defines dataframe to collect relevant info
voi = {'word':[], 'frequency':[],
        'return(24h)':[], 'volatility(24h)':[]}
data = pandas.DataFrame(voi)

#Define stopwords (not necessary but implemented for schooling purposes)
stop_words = get_stop_words('german')
stop_words.append('dpaafx')

#Loop through all articles to extract relevant words and allocate SMART sentiment weights
fault_counter = 0
for link in url_list:
    
    website = requests.get(link) 
    news = website.content 
    soup = BeautifulSoup(news, 'lxml')
    
    ###Data processing
    #Finanzen.net specific code to extract news' body
    container = soup.find('div', class_='teaser teaser-xs color-news')
    parts_html = list()
    for para in container.find_all('p', recursive=False):
        parts_html.append(para)
        
    #General text preparation code
    parts = remove_html_tags(str(parts_html))
    expression = "[^a-zA-Z äüöß]" 
    text_raw = re.sub(expression, '', str(parts))
    text_raw_lower = text_raw.lower()
    word_tokens = text_raw_lower.split()
    
    #Deleting stopwords from text is not necessary with SMART method but included for educational purpose
    filtered_sentence = list()
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w)
            
    #Finanzen.net specific code to extract stock's name
    name_section = soup.find('div', class_='chart-block relative')
    name_parts_html = list()
    for para in name_section.find_all('a', recursive=False):
        name_parts_html.append(para)
    name_parts = remove_html_tags(str(name_parts_html))
    name = name_parts[1:len(name_parts)-18]
    
    #Finanzen.net specific code to extract time of news
    date_section = soup.find(class_="pull-left mright-20")
    date = str(date_section)[33:49]
    date_time = datetime.datetime.strptime(date, '%d.%m.%Y %H:%M')
    
    #Gathers stock's return history
    try: #keeps code running if news date is faulty
        date = date_time.strftime('%Y-%m-%d')
        rounded_time = roundtime(date_time)
        t_end = (date_time + datetime.timedelta(days=3)).strftime('%Y-%m-%d')
        t_start = (date_time + datetime.timedelta(days=-5)).strftime('%Y-%m-%d')
        symbol = getCompany(name).get('symbol')
        stock_data = yfinance.Ticker(symbol)
        return_t = stock_data.history(start = t_start, end = t_end, interval = "5m")
        
        #Calculate return and volatility
        return_24 = 100*((((return_t.loc[(rounded_time).strftime('%Y-%m-%d %H:%M:%S') : t_end])["Open"].iloc[77])-
                          ((return_t.loc[t_start : (rounded_time).strftime('%Y-%m-%d %H:%M:%S')])["Open"].iloc[-1]))/((return_t.loc[t_start : (rounded_time).strftime('%Y-%m-%d %H:%M:%S')])["Open"].iloc[-1]))
        var_till_t = (numpy.var((return_t.loc[t_start : (rounded_time).strftime('%Y-%m-%d %H:%M:%S')])["Open"])) # variance until news
        var_after_t = (numpy.var((return_t.loc[(rounded_time).strftime('%Y-%m-%d %H:%M:%S') : t_end])["Open"])) # variance after news
        volatility_24 = math.sqrt(var_after_t)/math.sqrt(var_till_t)
        
        #Save in dict
        for word in filtered_sentence:
            if word in data.values:
                data.set_value(data['word'] == word, 'return(24h)', #addressed cell
                               round((((data.get_value(data.loc[data['word']==word].index[0], 2, takeable = True)*
                                 data.get_value(data.loc[data['word']==word].index[0], 1, takeable = True))+return_24)/
                                (data.get_value(data.loc[data['word']==word].index[0], 1, takeable = True)+1)), 4)) #change in cell
                data.set_value(data['word'] == word, 'volatility(24h)',
                               round((((data.get_value(data.loc[data['word']==word].index[0], 3, takeable = True)*
                                 data.get_value(data.loc[data['word']==word].index[0], 1, takeable = True))+volatility_24)/
                                (data.get_value(data.loc[data['word']==word].index[0], 1, takeable = True)+1)), 4))
                data.set_value(data['word'] == word, 'frequency',
                               data.get_value(data.loc[data['word']==word].index[0], 1, takeable = True)+1)
            else:
                data = data.append({'word':word, 'frequency':1,
                                    'return(24h)':return_24, 'volatility(24h)':volatility_24, }, ignore_index=True)
    except:
        fault_counter = fault_counter + 1

data = data.sort_values(by = 'word', ascending = True)
success_counter = len(url_list)-fault_counter

print('Hello ' + os.getlogin() + ',')
print('within the last minutes, ' + str(success_counter) + ' news articles could be assessed for training of the algorithm.')
print('Unfortunately, ' + str(fault_counter) + ' articles did not fulfill the requirements for assessment and were not included in the calculation.')
print('Among the most common reasons for exclusion are:')
print('-Publication was too recent to gather enough return data')
print('-Article was published on weekends and might be outdated by the time the stock exchange opens again')
print('-Article was not published on Finanzen.net directly but rather refers to another website')
print('-Publication date is outside requestable return range (60 days)')
print('Please refer to the Meta-Information for further explanations.')
print('However, we will continue with the valid data.')

data.to_csv(os.path.join(wd, 'data.csv'), index = False, header = True)
