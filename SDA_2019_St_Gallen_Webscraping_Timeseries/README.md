[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **SDA_2019_Machine_Learning_Asset_Allocation_ParsePriceData** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml


Name of Quantlet: 'SDA_2019_Machine_Learning_Asset_Allocation_ParsePriceData'

Published in: 'Statistical programming language Python - Student Project on "Machine Learning Asset Allocation"'

Description: 'Web scrape data for Crypto Assets and SP500 constituents from the internet. The tickers are scraped from wikipedia and timeseries data is downloaded.'

Keywords: 'web scraping, wikipedia, timeseries, price data'

Author: 'Julian Woessner'

See also: 'SDA_2019_St_Gallen_Hierarchical_Risk_Parity'

Submitted:  '13 November 2019'

Output:  'crypto_prices.csv, SP500_price_data_00.csv, SP500_price_data_15.csv, sp500tickers.pickle'

```

### PYTHON Code
```python

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:54:51 2019
This file contains all code to parse the closing prices of the SP500 
constituents from yahoo.finance
Code found on 
https://pythonprogramming.net/sp500-company-price-data-python-programming-for-finance
/?completed=/sp500-company-list-python-programming-for-finance/
Code adapted to be used for required data download.

@author: julianwossner
@date: 20191117
"""

# In[1]:
# Import packages
import bs4 as bs
import os
import pandas_datareader as data
import pickle
import requests



# In[2]:
# Define Function to parse tickers of SP500 companies

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    return tickers



# In[3]:
# Parse tickers and timeseries from 2015 to 2019, then save as csv
    
tickers = save_sp500_tickers()
tickers = [s.replace("\n","") for s in tickers]

# Define start and end data
start_date = '2015-01-01'
end_date =  '2019-09-30'

SP500_data = data.DataReader(tickers, 'yahoo', start_date, end_date)
SP500_close = SP500_data["Close"]  
 
# Save as .csv file
SP500_close.to_csv("SP500_price_data_15.csv")



# In[4]:
# Parse tickers and timeseries from 2015 to 2019, then save as csv
    
tickers = save_sp500_tickers()
tickers = [s.replace("\n","") for s in tickers]

# Define start and end data
start_date = '2000-01-01'
end_date =  '2019-09-30'

SP500_data = data.DataReader(tickers, 'yahoo', start_date, end_date)
SP500_close = SP500_data["Close"]   
# Save as .csv file
SP500_close.to_csv("SP500_price_data_00.csv")



# In[ ]

```

automatically created on 2019-11-18