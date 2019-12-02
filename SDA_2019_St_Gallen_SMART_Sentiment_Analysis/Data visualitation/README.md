[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **SDA_2019_St_Gallen_SMART_Sentiment_Analysis_Data_visualization** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet: SDA_2019_St_Gallen_SMART_Sentiment_Analysis_Data_visualization

Published in: Smart Data Analytics / University of St. Gallen (HSG)

Description: This Quantlet takes the output of the web scraping and forecasting Quantlets to visualize the results as a WordCloud and a scatterplot for the prediction vs realized return and volatility of the stock in the test sample.

Keywords: Data visualization, WordCloud, scatterplot, automatic sentiment analysis, stock-market news

Author: Oliver Kostorz

```

### PYTHON Code
```python

'''
This file visualizes the results obtained during the webscraping and forecasting part of the project.
Methods used are wordcloads, a html table and scatterplots.

Code was written by Oliver Kostorz during November 2019.
'''

#Import packages
import pandas
import numpy
import os
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image

#Set working device
try:
    wd = os.path.join(os.getcwd(), 'SDA-Oliver-Kostorz-SMART-Sentiment-Analysis-master')
except:
    print('Please specify path to working directiory manually.')
    
data = pandas.read_csv(os.path.join(wd, 'data.csv'))
evaluation = pandas.read_csv(os.path.join(wd, 'evaluation.csv'))


###Data visualisation

#Dictionary
data = data.sort_values(by = 'word', ascending = True)
data.to_html(os.path.join(wd, 'Dictionary.html'), justify = 'left', index = False)

#Wordcloud
wordcloud_top_return = data.sort_values(by = 'return(24h)', ascending = False).head(500)[['word']]
wordcloud_flop_return = data.sort_values(by = 'return(24h)', ascending = True).head(500)[['word']]

wordcloud_top_volatility = data.sort_values(by = 'volatility(24h)', ascending = True).head(500)[['word']]
wordcloud_flop_volatility = data.sort_values(by = 'volatility(24h)', ascending = False).head(500)[['word']]

words_top_return = str(wordcloud_top_return.values)
words_flop_return = str(wordcloud_flop_return.values)
words_top_volatility = str(wordcloud_top_volatility.values)
words_flop_volatility = str(wordcloud_flop_volatility.values)

flop = numpy.array(Image.open(os.path.join(wd, 'flop.jpg')))
top = numpy.array(Image.open(os.path.join(wd, 'top.jpg')))

wc_top_return = WordCloud(max_words = 500, mask = top, mode = 'RGBA', background_color = None).generate(words_top_return)
wc_top_return.to_file(os.path.join(wd, 'WordCloud_top_return.png'))

wc_flop_return = WordCloud(max_words = 500, mask = flop, mode = 'RGBA', background_color = None).generate(words_flop_return)
wc_flop_return.to_file(os.path.join(wd, 'WordCloud_flop_return.png'))

wc_top_volatility = WordCloud(max_words = 500, mask = top, mode = 'RGBA', background_color = None).generate(words_top_volatility)
wc_top_volatility.to_file(os.path.join(wd, 'WordCloud_top_volatility.png'))

wc_flop_volatility = WordCloud(max_words = 500, mask = flop, mode = 'RGBA', background_color = None).generate(words_flop_volatility)
wc_flop_volatility.to_file(os.path.join(wd, 'WordCloud_flop_volatility.png'))


#Scatterplot prediction-actual return
ret_forecast_eval = evaluation.plot(x = 'return realization', y = 'return prediction', style = 'o',
                                    title = 'Return forecast evaluation', legend = False,
                                    xlim = ([min(evaluation['return realization'].min(), evaluation['return prediction'].min())-1, max(evaluation['return realization'].max(), evaluation['return prediction'].max())+1]),
                                    ylim = ([min(evaluation['return realization'].min(), evaluation['return prediction'].min())-1, max(evaluation['return realization'].max(), evaluation['return prediction'].max())+1]))
plt.ylabel('return prediction')
ret_forecast_eval.figure.savefig(os.path.join(wd, 'ret_for_eva.png'), transperent = True)
img = Image.open(os.path.join(wd, 'ret_for_eva.png'))
img = img.convert("RGBA")
datas = img.getdata()
newData = []
for item in datas:
    if item[0] == 255 and item[1] == 255 and item[2] == 255:
        newData.append((255, 255, 255, 0))
    else:
        newData.append(item)
img.putdata(newData)
img.save(os.path.join(wd, 'ret_for_eva.png'), "PNG")

#Scatterplot prediction-actual volatility
vol_forecast_evaluation = evaluation.plot(x = 'volatility realization', y = 'volatility prediction', style = 'o',
                                          title = 'Volatility forecast evaluation', legend = False,
                                          xlim = ([min(evaluation['volatility realization'].min(), evaluation['volatility prediction'].min())-0.1, max(evaluation['volatility realization'].max(), evaluation['volatility prediction'].max())+0.1]),
                                          ylim = ([min(evaluation['volatility realization'].min(), evaluation['volatility prediction'].min())-0.1, max(evaluation['volatility realization'].max(), evaluation['volatility prediction'].max())+0.1]))
plt.ylabel('volatility prediction')
vol_forecast_evaluation.figure.savefig(os.path.join(wd, 'vol_for_eva.png'), transperent = True)
img = Image.open(os.path.join(wd, 'vol_for_eva.png'))
img = img.convert("RGBA")
datas = img.getdata()
newData = []
for item in datas:
    if item[0] == 255 and item[1] == 255 and item[2] == 255:
        newData.append((255, 255, 255, 0))
    else:
        newData.append(item)
img.putdata(newData)
img.save(os.path.join(wd, 'vol_for_eva.png'), "PNG")

```

automatically created on 2019-12-02