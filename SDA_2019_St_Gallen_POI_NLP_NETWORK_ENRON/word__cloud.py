# Source = class material

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import os
from os import path

def logo_word_enron():
    d = os.getcwd()
    text = open(path.join(d,"output.txt"),encoding = 'UTF8').read()
    stopwords = set(STOPWORDS)
    font_path=path.join(d,"msyh.ttf")
    mask = np.array(Image.open(path.join(d, "Enron.png")))
    wc = WordCloud(max_words=1000, mask=mask,
                   stopwords=stopwords,font_path=font_path, mode='RGBA', background_color=None)

    # Pass Text
    wc.generate(text)

    # to show the picture 
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")# 
    plt.show()



def logo_word_uni():
    
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud, STOPWORDS
    import os
    from os import path
    from matplotlib.pyplot import figure

    d = os.getcwd()
    text = open(path.join(d,"output.txt"),encoding = 'UTF8').read()
    stopwords = set(STOPWORDS)
    font_path=path.join(d,"msyh.ttf")
    mask = np.array(Image.open(path.join(d, "Uni.png")))
    wc = WordCloud(max_words=1000, mask=mask,
                   stopwords=stopwords,font_path=font_path, mode='RGBA', background_color=None)

    # Pass Text
    wc.generate(text)

    # to show the picture 
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()





