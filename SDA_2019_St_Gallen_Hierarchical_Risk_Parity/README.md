[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **SDA_2019_Machine_Learning_Asset_Allocation_RHP** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml


Name of Quantlet: 'SDA_2019_Machine_Learning_Asset_Allocation_RHP'

Published in: 'Statistical programming language Python - Student Project on "Machine Learning Asset Allocation"'

Description: 'Implementation of three different portfolio optimisation methods. The classical Markowitz Minimum Variance technique is compared to the Inverse Variance Portfolio and a new approach called Hierarchical Risk Parity (HRP). HRP combines graph theory and machine learning techniques to determine the optimal allocation to assets based on the information contained in the covariance matrix. Strategies are applied to Crypto Currency Assets and Stocks and evaluated on risk/return profile.'

Keywords: 'optimisation, portfolio allocation, asset allocation, machine learning, graph theory, hierarchical risk parity, markowitz, minimum variance, risk based investing'

Author: 'Julian Woessner'

See also: 'SDA_2019_St_Gallen_Webscraping_Timeseries'

Submitted:  '13 November 2019'

Input: 'Webscraped time series data for Crypto Currencies and SP500 constituents'

Output:  'Corr_Heatmap_Crypto_sorted.png, Corr_Heatmap_Crypto_unsorted.png, Corr_Heatmap_Mixed_sorted.png, Corr_Heatmap_Mixed_unsorted.png, Corr_Heatmap_SP500_15_sorted.png, Corr_Heatmap_SP500_unsorted.png, Corr_Network_Crypto_unsorted.png, Dendogram_Crypto.png, Dendogram_Mixed.png Dendogram_SP500.png, Index_crypto.png, Index_Mixed_16_30.png, Index_Mixed_16_90.png, Index_Mixed_16_180.png, Index_Mixed_16_360.png, Index_SP500_00_30.png, Index_SP500_00_90.png, Index_SP500_00_180.png, Index_SP500_00_360.png, Index_SP500_15_90.png'

```

### PYTHON Code
```python

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file implements the HRP algorithm, the MinVAr portfolio and the IVP portf.
for the fulfillment of the 2019 SDA class in St.Gallen, CH.
Code for HRP is based on Lopez de Prado, M. (2018). Advances in Financial 
Machine Learning. Wiley. The code has been adapted in order to be used with
python 3 and the data set.
 
@author: julianwossner
@date: 20191116
"""

# In[1]: 
# Load modules

import os
path = os.getcwd() # Set Working directory here

# Import modules for Datastructuring and calc.
import pandas as pd
import numpy as np
from scipy import stats
import warnings
from tqdm import tqdm

# Modules for RHP algorithm
import matplotlib.pyplot as mpl
import scipy.cluster.hierarchy as sch

# Modules for the network plot
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix

# Modules for Markowitz optimization
import cvxopt as opt
import cvxopt.solvers as optsolvers

warnings.filterwarnings("ignore") # suppress warnings in clustering



# In[2]: 
# define functions for HRP and IVP

def getIVP(cov,**kargs):
    # Compute the inverse-variance portfolio
    ivp=1./np.diag(cov)
    ivp/=ivp.sum()
    return ivp


def getClusterVar(cov, cItems):
    # Compute variance per cluster
    cov_=cov.loc[cItems, cItems] # matrix slice
    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar


def getQuasiDiag(link):
    # Sort clustered items by distance
    link=link.astype(int)
    sortIx=pd.Series([link[-1,0],link[-1,1]])
    numItems=link[-1,3] # number of original items
    while sortIx.max() >=numItems:
        sortIx.index=range(0,sortIx.shape[0]*2,2) # make space
        df0=sortIx[sortIx>=numItems] # find clusters
        i=df0.index;j=df0.values-numItems
        sortIx[i]=link[j,0] # item 1
        df0=pd.Series(link[j,1], index=i+1)
        sortIx=sortIx.append(df0) # item 2
        sortIx=sortIx.sort_index() # re-sort
        sortIx.index=range(sortIx.shape[0]) # re-index
    return sortIx.tolist()


def getRecBipart(cov,sortIx):
    # Compute HRP alloc
    w=pd.Series(1,index=sortIx)
    cItems=[sortIx] # initialize all items in one cluster
    while len(cItems)>0:
        cItems=[i[j:k] for i in cItems for j,k in ((0,len(i)//2),(len(i)//2,\
                len(i))) if len(i)>1] # bi-section
        for i in range(0,len(cItems),2): # parse in pairs
            cItems0=cItems[i] # cluster 1
            cItems1=cItems[i+1] # cluster 2
            cVar0=getClusterVar(cov,cItems0)
            cVar1=getClusterVar(cov,cItems1)
            alpha=1-cVar0/(cVar0+cVar1)
            w[cItems0]*=alpha # weight 1
            w[cItems1]*=1-alpha # weight 2
    return w


def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1 
    # This is a proper distance metric
    dist=((1-corr)/2.)**.5 # distance matrix
    return dist


def plotCorrMatrix(path, corr, labels=None):
    # Heatmap of the correlation matrix
    if labels is None:labels=[]
    mpl.pcolor(corr)
    mpl.colorbar()
    mpl.yticks(np.arange(.5,corr.shape[0]+.5),labels)
    mpl.xticks(np.arange(.5,corr.shape[0]+.5),labels)
    mpl.savefig(path,dpi=300, transparent=True)
    mpl.clf();mpl.close() # reset pylab
    return



# In[3]: 
# define function for MinVar portfolio
    
# The MIT License (MIT)
#
# Copyright (c) 2015 Christian Zielinski
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.   

def min_var_portfolio(cov_mat, allow_short=False):
    """
    Computes the minimum variance portfolio.

    Note: As the variance is not invariant with respect
    to leverage, it is not possible to construct non-trivial
    market neutral minimum variance portfolios. This is because
    the variance approaches zero with decreasing leverage,
    i.e. the market neutral portfolio with minimum variance
    is not invested at all.
    
    Parameters
    ----------
    cov_mat: pandas.DataFrame
        Covariance matrix of asset returns.
    allow_short: bool, optional
        If 'False' construct a long-only portfolio.
        If 'True' allow shorting, i.e. negative weights.

    Returns
    -------
    weights: pandas.Series
        Optimal asset weights.
    """
    if not isinstance(cov_mat, pd.DataFrame):
        raise ValueError("Covariance matrix is not a DataFrame")

    n = len(cov_mat)
        
    P = opt.matrix(cov_mat.values)
    q = opt.matrix(0.0, (n, 1))

# Constraints Gx <= h
    if not allow_short:
    # x >= 0
        G = opt.matrix(-np.identity(n))
        h = opt.matrix(0.0, (n, 1))
    else:
        G = None
        h = None

# Constraints Ax = b
# sum(x) = 1
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Solve
    optsolvers.options['show_progress'] = False
    sol = optsolvers.qp(P, q, G, h, A, b)
        
    if sol['status'] != 'optimal':
        warnings.warn("Convergence problem")
            
 # Put weights into a labeled series
    weights = pd.Series(sol['x'], index=cov_mat.index)
    return weights  



# In[4]: 
# Define functions for network graphs
    
#Function to plot Network plots
def plotNetwork(path,corr):
    # Transform it in a links data frame
    #links=corr.stack().reset_index()
    #Build graph
    corr=Corr_mat
    adj_matrix = corr
    constits_latest = corr.index
    # remove self-loops
    adj_matrix = np.where((adj_matrix<=1.000001) & (adj_matrix>=0.99999),0,adj_matrix)
    # replace values that are below threshold
    # create undirected graph from adj_matrix
    graph = from_numpy_matrix(adj_matrix, parallel_edges=False, create_using= nx.Graph())
    # set names to crypots
    graph = nx.relabel.relabel_nodes(graph, dict(zip(range(len(constits_latest)), constits_latest)))
    pos_og =  nx.circular_layout(graph, scale=2)
    pos = nx.circular_layout(graph, scale=1.7)
    
    for p in pos:  # raise text positions
        if pos[p][1]>1:
            pos[p][1] += 0.15
        if pos[p][1]<-1:
            pos[p][1] -= 0.15
        elif pos[p][0]<0:
            pos[p][0] -= 0.3
        else:
            pos[p][0]+=0.3
    plt = mpl.figure(figsize = (5,5)) 
    nx.draw(graph, pos_og, with_labels= False)
    nx.draw_networkx_labels(graph, pos)
     
    plt.savefig(path,dpi=300 ,transparent=True)
    mpl.clf();mpl.close()
    return



# In[5]:
# Loading and structuring crypto data sets

Crypto = pd.read_csv("crypto_prices.csv") #load csv
Crypto = Crypto[(~Crypto.isnull()).all(axis=1)] # Deleting empty rows

Crypto["date"] = Crypto['date'].map(lambda x: str(x)[:-9]) # Removing timestamp
Crypto = Crypto.rename(columns = {"date":"Date"})
Crypto = Crypto.replace(to_replace = 0, method = "ffill")
Price_data_univ=Crypto
#Price_data_univ = pd.merge(SP500, Crypto, on='Date', how='inner')#rename column
Price_data_univ = Price_data_univ.set_index("Date") # define Date  as index
# Calculating returns 
 


Return_data_univ = Price_data_univ.pct_change() #calculate daily returns
Return_data_univ = Return_data_univ.drop(Return_data_univ.index[range(0,1)])

# Calculating covariance matrix

Cov_mat = Return_data_univ.cov() # Covariance matrix of the return matrix
Corr_mat=Return_data_univ.corr() # Correlation matrix of the return matrix



# In[6]:
# Heatmap and network analysis of corr. matrix

# Plotting Correlation matrix heatmap

plotCorrMatrix(path+"/Corr_Heatmap_Crypto_unsorted",Corr_mat)


# network plot of correlation matrix

plotNetwork(path+"/Corr_Network_Crypto_unsorted.png", Corr_mat)

# Sort correlation matrix
dist=correlDist(Corr_mat)
link=sch.linkage(dist,'single')
sortIx=getQuasiDiag(link) 
sortIx=Corr_mat.index[sortIx].tolist() # recover labels 
Corr_sorted=Corr_mat.loc[sortIx,sortIx] # reorder

# Plot sorted correlation matrix
plotCorrMatrix(path+"/Corr_Heatmap_Crypto_sorted",Corr_sorted)


# Plot dendogram of the constituents
#2) Cluster Data
mpl.figure(num=None, figsize=(20, 10), dpi=300, facecolor='w', edgecolor='k')    
dn = sch.dendrogram(link, labels = dist.columns)
mpl.savefig(path+"/Dendrogram_Crypto.png", transparent = True, dpi = 300)
mpl.clf();mpl.close() # reset pylab



# In[7]:
#Function to calculate the HRP portfolio weights

def HRPportf(cov,corr):
    #1) Cluster covariance matrix
    dist=correlDist(corr)
    link=sch.linkage(dist,'single')
    sortIx=getQuasiDiag(link) 
    sortIx=corr.index[sortIx].tolist() # recover labels
    #2) Allocate capital according to HRP
    weights_hrp=getRecBipart(cov,sortIx)
    return weights_hrp



# In[8]:
# Compute the weights for the Markowitz MinVar and the HRP portfolio and the 
# IVP portfolio

w_HRP=np.array([HRPportf(Cov_mat,Corr_mat).index,HRPportf(Cov_mat,Corr_mat).round(3)])
w_HRP=pd.DataFrame(np.transpose(w_HRP))
w_HRP.columns = ["Asset","Weights HRP"]

w_MinVar= np.array([min_var_portfolio(Cov_mat).index,min_var_portfolio(Cov_mat).round(3)])
w_MinVar=pd.DataFrame(np.transpose(w_MinVar))
w_MinVar.columns = ["Asset","Weights MinVar"]

w_IVP= np.array([Cov_mat.index, getIVP(Cov_mat).round(3)])
w_IVP=pd.DataFrame(np.transpose(w_IVP))
w_IVP.columns = ["Asset","Weights IVP"]

Weights = pd.merge(w_MinVar,w_IVP,\
                   on="Asset", how = "inner")
Weights = pd.merge(Weights,w_HRP,\
                   on="Asset", how = "inner")

print(Weights.to_latex(index=False)) # Latex table output



# In[9]:
# Backtesting the three optimisation methods for Crypto dataset

# Function to calculate the weigths in sample and then test out of sample
def Backtest_Crypto(returns, rebal = 30): # rebal = 30 default rebalancing after 1 month
    nrows = len(returns.index)-rebal # Number of iterations without first set to train
    rets_train = returns[:rebal]
    
    
    cov,corr = rets_train.cov(), rets_train.corr()
    w_HRP=np.array([HRPportf(cov,corr).index,HRPportf(cov,corr)])
    w_HRP=pd.DataFrame(np.transpose(w_HRP))
    w_HRP.columns = ["Asset","Weights HRP"]

    w_MinVar= np.array([cov.index,min_var_portfolio(cov)])
    w_MinVar=pd.DataFrame(np.transpose(w_MinVar))
    w_MinVar.columns = ["Asset","Weights MinVar"]

    w_IVP= np.array([cov.index, getIVP(cov)])
    w_IVP=pd.DataFrame(np.transpose(w_IVP))
    w_IVP.columns = ["Asset","Weights IVP"]
                     
    Weights = pd.merge(w_MinVar, w_IVP, on="Asset", how = "inner")
    Weights = pd.merge(Weights,w_HRP, on="Asset", how = "inner")
    Weights = Weights.drop(Weights.columns[0],axis=1).to_numpy() 
    

    portf_return = pd.DataFrame(columns=["MinVar","IVP","HRP"], index = range(nrows))
    
    for i in tqdm(range(rebal,nrows+rebal)):
    
            if i>rebal and i<nrows-rebal and i % rebal == 0: # Check for rebalancing date
                rets_train = returns[i-rebal:i]
                cov,corr = rets_train.cov(), rets_train.corr()
                w_HRP=np.array([HRPportf(cov,corr).index,HRPportf(cov,corr)])
                w_HRP=pd.DataFrame(np.transpose(w_HRP))
                w_HRP.columns = ["Asset","Weights HRP"]
                
                w_MinVar= np.array([cov.index,min_var_portfolio(cov)])
                w_MinVar=pd.DataFrame(np.transpose(w_MinVar))
                w_MinVar.columns = ["Asset","Weights MinVar"]
            
                w_IVP= np.array([cov.index, getIVP(cov)])
                w_IVP=pd.DataFrame(np.transpose(w_IVP))
                w_IVP.columns = ["Asset","Weights IVP"]
                     
                Weights = pd.merge(w_MinVar, w_IVP, on="Asset", how = "inner")
                Weights = pd.merge(Weights,w_HRP, on="Asset", how = "inner")
                Weights = Weights.drop(Weights.columns[0],axis=1).to_numpy()     
       
            

            portf_return["MinVar"][i-int(rebal):i-int(rebal)+1]=np.average(returns[i:i+1].to_numpy(),\
                    weights = Weights[:,0].reshape(len(returns.columns),1).ravel(), axis = 1)
            portf_return["IVP"][i-int(rebal):i-int(rebal)+1]=np.average(returns[i:i+1].to_numpy(),\
                    weights = Weights[:,1].reshape(len(returns.columns),1).ravel(), axis = 1)
            portf_return["HRP"][i-int(rebal):i-int(rebal)+1]=np.average(returns[i:i+1].to_numpy(),\
                    weights = Weights[:,2].reshape(len(returns.columns),1).ravel(), axis = 1)
           
            
    return portf_return



# In[10]:    
# Calculate the backtested portfolio returns
    
portf_rets = Backtest_Crypto(Return_data_univ, rebal=90)
portf_rets2 = 1+portf_rets

rebal = 90
# Calculate index
portf_index = pd.DataFrame(columns=["MinVar","IVP","HRP"], \
                           index = Return_data_univ.index)[rebal:]
portf_index[0:1] = 100
for i in range(1,len(portf_index.index)):
    portf_index["MinVar"][i:i+1] = float(portf_index["MinVar"][i-1:i]) * float(portf_rets2["MinVar"][i-1:i])
    portf_index["IVP"][i:i+1] = float(portf_index["IVP"][i-1:i]) * float(portf_rets2["IVP"][i-1:i])
    portf_index["HRP"][i:i+1] = float(portf_index["HRP"][i-1:i]) * float(portf_rets2["HRP"][i-1:i])

index = portf_index.plot.line()
mpl.savefig(path+"/Index_crypto.png", transparent = True, dpi = 300)
mpl.clf();mpl.close() # reset pylab


# Calculate portfolio return and portfolio variance
mean_MinVar = stats.mstats.gmean(np.array(portf_rets2["MinVar"], dtype=float))-1
mean_IVP = stats.mstats.gmean(np.array(portf_rets2["IVP"], dtype=float))-1
mean_HRP = stats.mstats.gmean(np.array(portf_rets2["HRP"], dtype=float))-1

# Daily Standard deviation
std_MinVar = portf_rets["MinVar"].std()
std_IVP = portf_rets["IVP"].std()
std_HRP = portf_rets["HRP"].std()

# Sharpe ratios
SR_MinVar = mean_MinVar/std_MinVar
SR_IVP = mean_IVP/std_IVP
SR_HRP = mean_HRP/std_HRP

Perf_figures = pd.DataFrame([[mean_MinVar, mean_IVP,mean_HRP], [std_MinVar, \
                            std_IVP, std_HRP],[SR_MinVar,SR_IVP,SR_HRP]], \
    index =['Mean', 'Std', 'SR'], columns = ["MinVar","IVP","HRP"])

print(Perf_figures.to_latex(index=True)) # Latex table output



# In[11]:
# Analysis for the SP500 universe from 2015 onwards
# Data import

SP500 = pd.read_csv("SP500_price_data_15.csv") #load csv

#Deleting empty columns
SP500 = SP500.replace(to_replace = 0, method = "ffill")
Price_data_univ = SP500
Price_data_univ = Price_data_univ.set_index("Date") # define Date as index
# Drop columns with incomplete data
Price_data_univ = Price_data_univ.drop(["AMCR","BF.B","BKR","BRK.B","CTVA", \
                                        "DOW","FOX","FOXA","FTV","HPE","KHC", \
                                        "LW","PYPL","UA","WRK"], axis=1)

# Calculating returns and deleting columns that contain 0
Return_data_univ = Price_data_univ.pct_change() #calculate daily returns
Return_data_univ = Return_data_univ.drop(Return_data_univ.index[range(0,1)])
#Return_data_univ = Return_data_univ.fillna(0)

# Calculating covariance matrix
Cov_mat = Return_data_univ.cov() # Covariance matrix of the return matrix
Corr_mat=Return_data_univ.corr() # Correlation matrix of the return matrix



# In[12]:
# Plotting Correlation matrix heatmap

plotCorrMatrix(path+"/Corr_Heatmap_SP500_15_unsorted",Corr_mat)

# Sort correlation matrix
dist=correlDist(Corr_mat)
link=sch.linkage(dist,'single')
sortIx=getQuasiDiag(link) 
sortIx=Corr_mat.index[sortIx].tolist() # recover labels 
Corr_sorted=Corr_mat.loc[sortIx,sortIx] # reorder

# Plot sorted correlation matrix
plotCorrMatrix(path+"/Corr_Heatmap_SP500_15_sorted",Corr_sorted)


# Plot dendogram of the constituents
#2) Cluster Data
mpl.figure(num=None, figsize=(20, 10), dpi=300, facecolor='w', edgecolor='k')    
dn = sch.dendrogram(link, labels = dist.columns, leaf_rotation=90., \
    leaf_font_size=8.) # font size for the x axis labels)
mpl.savefig(path+"/Dendrogram_SP500.png", transparent = True, dpi = 1200)
mpl.clf();mpl.close() # reset pylab



# In[13]:
# Backtesting function for stocks data
#
def Backtest_SP500(returns, rebal = 30): # rebal = 30 default rebalancing after 1 month
    nrows = len(returns.index)-rebal # Number of iterations without first set to train
    rets_train = returns[:rebal]
    
    cov,corr = rets_train.cov(), rets_train.corr()
    w_HRP=np.array([HRPportf(cov,corr).index,HRPportf(cov,corr)])
    w_HRP=pd.DataFrame(np.transpose(w_HRP))
    w_HRP.columns = ["Asset","Weights HRP"]

    w_MinVar= np.array([cov.index,min_var_portfolio(cov)])
    w_MinVar=pd.DataFrame(np.transpose(w_MinVar))
    w_MinVar.columns = ["Asset","Weights MinVar"]

    w_IVP= np.array([cov.index, getIVP(cov)])
    w_IVP=pd.DataFrame(np.transpose(w_IVP))
    w_IVP.columns = ["Asset","Weights IVP"]
                     
    Weights = pd.merge(w_MinVar, w_IVP, on="Asset", how = "inner")
    Weights = pd.merge(Weights,w_HRP, on="Asset", how = "inner")
    Weights = Weights.drop(Weights.columns[0],axis=1).to_numpy() 
    

    portf_return = pd.DataFrame(columns=["MinVar","IVP","HRP"], index = range(nrows))
    
    for i in tqdm(range(rebal,nrows+rebal)):
           
        
            if i>rebal and i<nrows-rebal and i % rebal == 0: # Check for rebalancing date
                rets_train = returns[i-rebal:i]
                cov,corr = rets_train.cov(), rets_train.corr()
                w_HRP=np.array([HRPportf(cov,corr).index,HRPportf(cov,corr)])
                w_HRP=pd.DataFrame(np.transpose(w_HRP))
                w_HRP.columns = ["Asset","Weights HRP"]
                
                w_MinVar= np.array([cov.index,min_var_portfolio(cov)])
                w_MinVar=pd.DataFrame(np.transpose(w_MinVar))
                w_MinVar.columns = ["Asset","Weights MinVar"]
            
                w_IVP= np.array([cov.index, getIVP(cov)])
                w_IVP=pd.DataFrame(np.transpose(w_IVP))
                w_IVP.columns = ["Asset","Weights IVP"]
                     
                Weights = pd.merge(w_MinVar, w_IVP, on="Asset", how = "inner")
                Weights = pd.merge(Weights,w_HRP, on="Asset", how = "inner")
                Weights = Weights.drop(Weights.columns[0],axis=1).to_numpy()     
       
            

            portf_return["MinVar"][i-int(rebal):i-int(rebal)+1]=np.average(returns[i:i+1].to_numpy(),\
                    weights = Weights[:,0].reshape(len(returns.columns),1).ravel(), axis = 1)
            portf_return["IVP"][i-int(rebal):i-int(rebal)+1]=np.average(returns[i:i+1].to_numpy(),\
                    weights = Weights[:,1].reshape(len(returns.columns),1).ravel(), axis = 1)
            portf_return["HRP"][i-int(rebal):i-int(rebal)+1]=np.average(returns[i:i+1].to_numpy(),\
                    weights = Weights[:,2].reshape(len(returns.columns),1).ravel(), axis = 1)
            
    return portf_return



# In[14]:  
# Calculate the backtested portfolio returns
    
portf_rets_15 = Backtest_SP500(Return_data_univ, rebal=90) # Change rebalancing frequency
portf_rets_2 = 1+portf_rets_15

rebal = 90
# Calculate index
portf_index = pd.DataFrame(columns=["MinVar","IVP","HRP"], \
                           index = Return_data_univ.index)[rebal:]
portf_index[0:1] = 100
for i in range(1,len(portf_index.index)):
    portf_index["MinVar"][i:i+1] = float(portf_index["MinVar"][i-1:i]) * float(portf_rets_2["MinVar"][i-1:i])
    portf_index["IVP"][i:i+1] = float(portf_index["IVP"][i-1:i]) * float(portf_rets_2["IVP"][i-1:i])
    portf_index["HRP"][i:i+1] = float(portf_index["HRP"][i-1:i]) * float(portf_rets_2["HRP"][i-1:i])

index = portf_index.plot.line()
mpl.savefig(path+"/Index_SP500_15_90.png", transparent = True, dpi = 300)


# Calculate portfolio return and portfolio variance
# Daily average return
mean_MinVar = stats.mstats.gmean(np.array(portf_rets_2["MinVar"], dtype=float))-1
mean_IVP = stats.mstats.gmean(np.array(portf_rets_2["IVP"], dtype=float))-1
mean_HRP = stats.mstats.gmean(np.array(portf_rets_2["HRP"], dtype=float))-1
# Daily Standard deviation
std_MinVar = portf_rets_15["MinVar"].std()
std_IVP = portf_rets_15["IVP"].std()
std_HRP = portf_rets_15["HRP"].std()
# Sharpe ratios
SR_MinVar = mean_MinVar/std_MinVar
SR_IVP = mean_IVP/std_IVP
SR_HRP = mean_HRP/std_HRP

Perf_figures = pd.DataFrame([[mean_MinVar, mean_IVP,mean_HRP], [std_MinVar, \
                            std_IVP, std_HRP],[SR_MinVar,SR_IVP,SR_HRP]], \
    index =['Mean', 'Std', 'SR'], columns = ["MinVar","IVP","HRP"])

print(Perf_figures.to_latex(index=True)) # Latex table output



# In[15]:
# Analysis for the SP500 universe from 2000 onwards: Data import

SP500 = pd.read_csv("SP500_price_data_00.csv") #load csv
#Deleting empty columns
#SP500 = SP500[[col for col in SP500.columns if SP500.loc[:,col].notna().any()]] 
SP500 = SP500.replace(to_replace = 0, method = "ffill")
Price_data_univ = SP500
Price_data_univ = Price_data_univ.set_index("Date") # define Date  as index
Price_data_univ = Price_data_univ.drop(["AAL","AAP","ABBV","ACN","ADS","AIZ",\
                                        "ALGN","ALLE","AMCR","AMP","ANET","ANTM"\
                                        ,"APTV","AVGO","AWK","BF.B","BKR","BR",\
                                        "BRK.B","CBOE","CBRE","CBS","CDW","CE",\
                                        "CFG","CHTR","CME","CMG"	,"CNC","COTY",\
                                        "CPRI","CTVA","CXO"	,"DAL","DFS","DG", \
                                        "DISCA", "DISCK","DLR","DOW","EQIX",\
                                        "EW","EXPE","EXR","FANG","FB","FBHS",\
                                        "FLT","FOX","FOXA","FRC","FTI","FTNT",\
                                        "FTV","GM","GOOG","GOOGL","GPN","GRMN",\
                                        "HBI","HCA","HII","HLT","HPE","ICE",\
                                        "ILMN","INFO","IPGP","IQV","ISRG",\
                                        "KEYS","KMI","LDOS","LKQ","LVS","LW",\
                                        "MA","MDLZ","MET","MKTX","MPC","MSCI",\
                                        "NCLH", "NDAQ","NFLX", "NLSN","NRG",\
                                        "NWSA", "PFG", "PKG"	,"PM","PRU","PSX",\
                                        "PYPL","QRVO","STX","SYF","TDG","TEL",\
                                        "TMUS","TPR","TRIP",	"TWTR","UA","UAA",\
                                        "UAL","ULTA","V","VIAB","VRSK","WCG",\
                                        "WLTW","WRK"	, "WU",	"WYNN","XEC"	,"XYL",\
                                        "ZBH","ZTS","CF","CRM","FIS","KHC","LYB",\
                                        "NWS"], axis=1)

# Calculating returns and deleting columns that contain 0
Return_data_univ = Price_data_univ.pct_change() #calculate daily returns
Return_data_univ = Return_data_univ.drop(Return_data_univ.index[range(0,1)])


# Calculating covariance matrix
Cov_mat = Return_data_univ.cov() # Covariance matrix of the return matrix
Corr_mat=Return_data_univ.corr() # Correlation matrix of the return matrix



# In[16]:
# Calculate the backtested portfolio returns

portf_rets = Backtest_SP500(Return_data_univ, rebal=90) # Change rebalancing frequency
portf_rets2 = 1+portf_rets

rebal = 90
# Calculate index
portf_index = pd.DataFrame(columns=["MinVar","IVP","HRP"], \
                           index = Return_data_univ.index)[rebal:]
portf_index[0:1] = 100
for i in range(1,len(portf_index.index)):
    portf_index["MinVar"][i:i+1] = float(portf_index["MinVar"][i-1:i]) * float(portf_rets2["MinVar"][i-1:i])
    portf_index["IVP"][i:i+1] = float(portf_index["IVP"][i-1:i]) * float(portf_rets2["IVP"][i-1:i])
    portf_index["HRP"][i:i+1] = float(portf_index["HRP"][i-1:i]) * float(portf_rets2["HRP"][i-1:i])

index = portf_index.plot.line()
mpl.savefig(path+"/Index_SP500_00_90.png", transparent = True, dpi = 300)


# Calculate portfolio return and portfolio variance
# Daily average return
mean_MinVar = stats.mstats.gmean(np.array(portf_rets2["MinVar"], dtype=float))-1
mean_IVP = stats.mstats.gmean(np.array(portf_rets2["IVP"], dtype=float))-1
mean_HRP = stats.mstats.gmean(np.array(portf_rets2["HRP"], dtype=float))-1
# Daily Standard deviation
std_MinVar = portf_rets["MinVar"].std()
std_IVP = portf_rets["IVP"].std()
std_HRP = portf_rets["HRP"].std()
# Sharpe ratios
SR_MinVar = mean_MinVar/std_MinVar
SR_IVP = mean_IVP/std_IVP
SR_HRP = mean_HRP/std_HRP

Perf_figures = pd.DataFrame([[mean_MinVar, mean_IVP,mean_HRP], [std_MinVar, \
                            std_IVP, std_HRP],[SR_MinVar,SR_IVP,SR_HRP]], \
    index =['Mean', 'Std', 'SR'], columns = ["MinVar","IVP","HRP"])

print(Perf_figures.to_latex(index=True)) # Latex table output



# In[17]:
# Changing the rebalancing frequency

portf_rets_30 = Backtest_SP500(Return_data_univ, rebal=30) # Monthly
portf_rets_30_2 = 1+portf_rets_30
portf_rets_180 = Backtest_SP500(Return_data_univ, rebal=180) # half-year
portf_rets_180_2 = 1+portf_rets_180
portf_rets_360 = Backtest_SP500(Return_data_univ, rebal=360) # yearly
portf_rets_360_2 = 1+portf_rets_360

#Print indices
rebal = 30
# Calculate index
portf_index = pd.DataFrame(columns=["MinVar","IVP","HRP"], \
                           index = Return_data_univ.index)[rebal:]
portf_index[0:1] = 100
for i in range(1,len(portf_index.index)):
    portf_index["MinVar"][i:i+1] = float(portf_index["MinVar"][i-1:i]) * float(portf_rets_30_2["MinVar"][i-1:i])
    portf_index["IVP"][i:i+1] = float(portf_index["IVP"][i-1:i]) * float(portf_rets_30_2["IVP"][i-1:i])
    portf_index["HRP"][i:i+1] = float(portf_index["HRP"][i-1:i]) * float(portf_rets_30_2["HRP"][i-1:i])

index = portf_index.plot.line()
mpl.savefig(path+"/Index_SP500_00_30.png", transparent = True, dpi = 300)


rebal = 180
# Calculate index
portf_index = pd.DataFrame(columns=["MinVar","IVP","HRP"], \
                           index = Return_data_univ.index)[rebal:]
portf_index[0:1] = 100
for i in range(1,len(portf_index.index)):
    portf_index["MinVar"][i:i+1] = float(portf_index["MinVar"][i-1:i]) * float(portf_rets_180_2["MinVar"][i-1:i])
    portf_index["IVP"][i:i+1] = float(portf_index["IVP"][i-1:i]) * float(portf_rets_180_2["IVP"][i-1:i])
    portf_index["HRP"][i:i+1] = float(portf_index["HRP"][i-1:i]) * float(portf_rets_180_2["HRP"][i-1:i])

index = portf_index.plot.line()
mpl.savefig(path+"/Index_SP500_00_180.png", transparent = True, dpi = 300)

rebal = 360
# Calculate index
portf_index = pd.DataFrame(columns=["MinVar","IVP","HRP"], \
                           index = Return_data_univ.index)[rebal:]
portf_index[0:1] = 100
for i in range(1,len(portf_index.index)):
    portf_index["MinVar"][i:i+1] = float(portf_index["MinVar"][i-1:i]) * float(portf_rets_360_2["MinVar"][i-1:i])
    portf_index["IVP"][i:i+1] = float(portf_index["IVP"][i-1:i]) * float(portf_rets_360_2["IVP"][i-1:i])
    portf_index["HRP"][i:i+1] = float(portf_index["HRP"][i-1:i]) * float(portf_rets_360_2["HRP"][i-1:i])

index = portf_index.plot.line()
mpl.savefig(path+"/Index_SP500_00_360.png", transparent = True, dpi = 300)


# Calculate performance figures

mean_MinVar = stats.mstats.gmean(np.array(portf_rets_30_2["MinVar"], dtype=float))-1
mean_IVP = stats.mstats.gmean(np.array(portf_rets_30_2["IVP"], dtype=float))-1
mean_HRP = stats.mstats.gmean(np.array(portf_rets_30_2["HRP"], dtype=float))-1

# Daily Standard deviation
std_MinVar = portf_rets_30["MinVar"].std()
std_IVP = portf_rets_30["IVP"].std()
std_HRP = portf_rets_30["HRP"].std()

# Sharpe ratios
SR_MinVar = mean_MinVar/std_MinVar
SR_IVP = mean_IVP/std_IVP
SR_HRP = mean_HRP/std_HRP

Perf_figures = pd.DataFrame([[mean_MinVar, mean_IVP,mean_HRP], [std_MinVar, \
                            std_IVP, std_HRP],[SR_MinVar,SR_IVP,SR_HRP]], \
    index =['Mean', 'Std', 'SR'], columns = ["MinVar","IVP","HRP"])

print(Perf_figures.to_latex(index=True)) # Latex table output

# Calculate performance figures

mean_MinVar = stats.mstats.gmean(np.array(portf_rets_180_2["MinVar"], dtype=float))-1
mean_IVP = stats.mstats.gmean(np.array(portf_rets_180_2["IVP"], dtype=float))-1
mean_HRP = stats.mstats.gmean(np.array(portf_rets_180_2["HRP"], dtype=float))-1

# Daily Standard deviation
std_MinVar = portf_rets_180["MinVar"].std()
std_IVP = portf_rets_180["IVP"].std()
std_HRP = portf_rets_180["HRP"].std()

# Sharpe ratios
SR_MinVar = mean_MinVar/std_MinVar
SR_IVP = mean_IVP/std_IVP
SR_HRP = mean_HRP/std_HRP

Perf_figures = pd.DataFrame([[mean_MinVar, mean_IVP,mean_HRP], [std_MinVar, \
                            std_IVP, std_HRP],[SR_MinVar,SR_IVP,SR_HRP]], \
    index =['Mean', 'Std', 'SR'], columns = ["MinVar","IVP","HRP"])

print(Perf_figures.to_latex(index=True)) # Latex table output

# Calculate performance figures

mean_MinVar = stats.mstats.gmean(np.array(portf_rets_360_2["MinVar"], dtype=float))-1
mean_IVP = stats.mstats.gmean(np.array(portf_rets_360_2["IVP"], dtype=float))-1
mean_HRP = stats.mstats.gmean(np.array(portf_rets_360_2["HRP"], dtype=float))-1

# Daily Standard deviation
std_MinVar = portf_rets_360["MinVar"].std()
std_IVP = portf_rets_360["IVP"].std()
std_HRP = portf_rets_360["HRP"].std()

# Sharpe ratios
SR_MinVar = mean_MinVar/std_MinVar
SR_IVP = mean_IVP/std_IVP
SR_HRP = mean_HRP/std_HRP

Perf_figures = pd.DataFrame([[mean_MinVar, mean_IVP,mean_HRP], [std_MinVar, \
                            std_IVP, std_HRP],[SR_MinVar,SR_IVP,SR_HRP]], \
    index =['Mean', 'Std', 'SR'], columns = ["MinVar","IVP","HRP"])

print(Perf_figures.to_latex(index=True)) # Latex table output



# In[18]:
# Create Mixed portfolios with stocks and crypto
# Data import

SP500 = pd.read_csv("SP500_price_data_00.csv") #load csv
SP500 = SP500.drop(["AAL","AAP","ABBV","ACN","ADS","AIZ",\
                                        "ALGN","ALLE","AMCR","AMP","ANET","ANTM"\
                                        ,"APTV","AVGO","AWK","BF.B","BKR","BR",\
                                        "BRK.B","CBOE","CBRE","CBS","CDW","CE",\
                                        "CFG","CHTR","CME","CMG"	,"CNC","COTY",\
                                        "CPRI","CTVA","CXO"	,"DAL","DFS","DG", \
                                        "DISCA", "DISCK","DLR","DOW","EQIX",\
                                        "EW","EXPE","EXR","FANG","FB","FBHS",\
                                        "FLT","FOX","FOXA","FRC","FTI","FTNT",\
                                        "FTV","GM","GOOG","GOOGL","GPN","GRMN",\
                                        "HBI","HCA","HII","HLT","HPE","ICE",\
                                        "ILMN","INFO","IPGP","IQV","ISRG",\
                                        "KEYS","KMI","LDOS","LKQ","LVS","LW",\
                                        "MA","MDLZ","MET","MKTX","MPC","MSCI",\
                                        "NCLH", "NDAQ","NFLX", "NLSN","NRG",\
                                        "NWSA", "PFG", "PKG"	,"PM","PRU","PSX",\
                                        "PYPL","QRVO","STX","SYF","TDG","TEL",\
                                        "TMUS","TPR","TRIP",	"TWTR","UA","UAA",\
                                        "UAL","ULTA","V","VIAB","VRSK","WCG",\
                                        "WLTW","WRK"	, "WU",	"WYNN","XEC"	,"XYL",\
                                        "ZBH","ZTS","CF","CRM","FIS","KHC","LYB",\
                                        "NWS"], axis=1)

SP500 = SP500.set_index("Date")
SP500_sample = SP500.sample(10,axis = 1,random_state=1) # randomly select 10 stocks from SP500

Price_data_univ2 = pd.merge(SP500_sample, Crypto, on='Date', how='inner')#rename column
Price_data_univ2 = Price_data_univ2.set_index("Date") # define Date  as index


# Calculating returns and deleting columns that contain 0
Return_data_univ2 = Price_data_univ2.pct_change() #calculate daily returns
Return_data_univ2 = Return_data_univ2.drop(Return_data_univ2.index[range(0,1)])


# Calculating covariance matrix
Cov_mat2 = Return_data_univ2.cov() # Covariance matrix of the return matrix
Corr_mat2=Return_data_univ2.corr() # Correlation matrix of the return matrix



# In[19]:

# Plotting Correlation matrix heatmap
plotCorrMatrix(path+"/Corr_Heatmap_Mixed_unsorted",Corr_mat2)

# Sort correlation matrix
dist=correlDist(Corr_mat2)
link=sch.linkage(dist,'single')
sortIx=getQuasiDiag(link) 
sortIx=Corr_mat2.index[sortIx].tolist() # recover labels 
Corr_sorted2=Corr_mat2.loc[sortIx,sortIx] # reorder

# Plot sorted correlation matrix
plotCorrMatrix(path+"/Corr_Heatmap_Mixed_sorted",Corr_sorted2)


# Plot dendogram of the constituents
# Cluster Data
mpl.figure(num=None, figsize=(20, 10), dpi=300, facecolor='w', edgecolor='k')    
dn = sch.dendrogram(link, labels = dist.columns)
mpl.savefig(path+"/Dendrogram_Mixed.png", transparent = True, dpi = 300)
mpl.clf();mpl.close() # reset pylab



# In[20]:
# Calculate the backtested portfolio returns
# Changing the rebalancing frequency

portf_rets_30 = Backtest_SP500(Return_data_univ2, rebal=30) # Monthly
portf_rets_30_2 = 1+portf_rets_30
portf_rets_180 = Backtest_SP500(Return_data_univ2, rebal=180) # half-year
portf_rets_180_2 = 1+portf_rets_180
portf_rets_360 = Backtest_SP500(Return_data_univ2, rebal=360) # yearly
portf_rets_360_2 = 1+portf_rets_360
portf_rets_90 = Backtest_SP500(Return_data_univ2, rebal=90) # Monthly
portf_rets_90_2 = 1+portf_rets_90

#Print indices
rebal = 30
# Calculate index
portf_index = pd.DataFrame(columns=["MinVar","IVP","HRP"], \
                           index = Return_data_univ2.index)[rebal:]
portf_index[0:1] = 100
for i in range(1,len(portf_index.index)):
    portf_index["MinVar"][i:i+1] = float(portf_index["MinVar"][i-1:i]) * float(portf_rets_30_2["MinVar"][i-1:i])
    portf_index["IVP"][i:i+1] = float(portf_index["IVP"][i-1:i]) * float(portf_rets_30_2["IVP"][i-1:i])
    portf_index["HRP"][i:i+1] = float(portf_index["HRP"][i-1:i]) * float(portf_rets_30_2["HRP"][i-1:i])

index = portf_index.plot.line( rot=25)
mpl.savefig(path+"/Index_Mixed_16_30.png", transparent = True, dpi = 300)


rebal = 180
# Calculate index
portf_index = pd.DataFrame(columns=["MinVar","IVP","HRP"], \
                           index = Return_data_univ2.index)[rebal:]
portf_index[0:1] = 100
for i in range(1,len(portf_index.index)):
    portf_index["MinVar"][i:i+1] = float(portf_index["MinVar"][i-1:i]) * float(portf_rets_180_2["MinVar"][i-1:i])
    portf_index["IVP"][i:i+1] = float(portf_index["IVP"][i-1:i]) * float(portf_rets_180_2["IVP"][i-1:i])
    portf_index["HRP"][i:i+1] = float(portf_index["HRP"][i-1:i]) * float(portf_rets_180_2["HRP"][i-1:i])

index = portf_index.plot.line(rot=25)
mpl.savefig(path+"/Index_Mixed_16_180.png", transparent = True, dpi = 300)


rebal = 360
# Calculate index
portf_index = pd.DataFrame(columns=["MinVar","IVP","HRP"], \
                           index = Return_data_univ2.index)[rebal:]
portf_index[0:1] = 100
for i in range(1,len(portf_index.index)):
    portf_index["MinVar"][i:i+1] = float(portf_index["MinVar"][i-1:i]) * float(portf_rets_360_2["MinVar"][i-1:i])
    portf_index["IVP"][i:i+1] = float(portf_index["IVP"][i-1:i]) * float(portf_rets_360_2["IVP"][i-1:i])
    portf_index["HRP"][i:i+1] = float(portf_index["HRP"][i-1:i]) * float(portf_rets_360_2["HRP"][i-1:i])

index = portf_index.plot.line(rot=25)
mpl.savefig(path+"/Index_Mixed_16_360.png", transparent = True, dpi = 300)


rebal = 90
# Calculate index
portf_index = pd.DataFrame(columns=["MinVar","IVP","HRP"], \
                           index = Return_data_univ2.index)[rebal:]
portf_index[0:1] = 100
for i in range(1,len(portf_index.index)):
    portf_index["MinVar"][i:i+1] = float(portf_index["MinVar"][i-1:i]) * float(portf_rets_90_2["MinVar"][i-1:i])
    portf_index["IVP"][i:i+1] = float(portf_index["IVP"][i-1:i]) * float(portf_rets_90_2["IVP"][i-1:i])
    portf_index["HRP"][i:i+1] = float(portf_index["HRP"][i-1:i]) * float(portf_rets_90_2["HRP"][i-1:i])

index = portf_index.plot.line( rot=25)
mpl.savefig(path+"/Index_Mixed_16_90.png", transparent = True, dpi = 300)


# Calculate performance figures

mean_MinVar = stats.mstats.gmean(np.array(portf_rets_30_2["MinVar"], dtype=float))-1
mean_IVP = stats.mstats.gmean(np.array(portf_rets_30_2["IVP"], dtype=float))-1
mean_HRP = stats.mstats.gmean(np.array(portf_rets_30_2["HRP"], dtype=float))-1
# Daily Standard deviation
std_MinVar = portf_rets_30["MinVar"].std()
std_IVP = portf_rets_30["IVP"].std()
std_HRP = portf_rets_30["HRP"].std()
# Sharpe ratios
SR_MinVar = mean_MinVar/std_MinVar
SR_IVP = mean_IVP/std_IVP
SR_HRP = mean_HRP/std_HRP

Perf_figures = pd.DataFrame([[mean_MinVar, mean_IVP,mean_HRP], [std_MinVar, \
                            std_IVP, std_HRP],[SR_MinVar,SR_IVP,SR_HRP]], \
    index =['Mean', 'Std', 'SR'], columns = ["MinVar","IVP","HRP"])

print(Perf_figures.to_latex(index=True)) # Latex table output

# Calculate performance figures

mean_MinVar = stats.mstats.gmean(np.array(portf_rets_180_2["MinVar"], dtype=float))-1
mean_IVP = stats.mstats.gmean(np.array(portf_rets_180_2["IVP"], dtype=float))-1
mean_HRP = stats.mstats.gmean(np.array(portf_rets_180_2["HRP"], dtype=float))-1

# Daily Standard deviation
std_MinVar = portf_rets_180["MinVar"].std()
std_IVP = portf_rets_180["IVP"].std()
std_HRP = portf_rets_180["HRP"].std()

# Sharpe ratios
SR_MinVar = mean_MinVar/std_MinVar
SR_IVP = mean_IVP/std_IVP
SR_HRP = mean_HRP/std_HRP

Perf_figures = pd.DataFrame([[mean_MinVar, mean_IVP,mean_HRP], [std_MinVar, \
                            std_IVP, std_HRP],[SR_MinVar,SR_IVP,SR_HRP]], \
    index =['Mean', 'Std', 'SR'], columns = ["MinVar","IVP","HRP"])

print(Perf_figures.to_latex(index=True)) # Latex table output

# Calculate performance figures

mean_MinVar = stats.mstats.gmean(np.array(portf_rets_360_2["MinVar"], dtype=float))-1
mean_IVP = stats.mstats.gmean(np.array(portf_rets_360_2["IVP"], dtype=float))-1
mean_HRP = stats.mstats.gmean(np.array(portf_rets_360_2["HRP"], dtype=float))-1

# Daily Standard deviation
std_MinVar = portf_rets_360["MinVar"].std()
std_IVP = portf_rets_360["IVP"].std()
std_HRP = portf_rets_360["HRP"].std()

# Sharpe ratios
SR_MinVar = mean_MinVar/std_MinVar
SR_IVP = mean_IVP/std_IVP
SR_HRP = mean_HRP/std_HRP

Perf_figures = pd.DataFrame([[mean_MinVar, mean_IVP,mean_HRP], [std_MinVar, \
                            std_IVP, std_HRP],[SR_MinVar,SR_IVP,SR_HRP]], \
    index =['Mean', 'Std', 'SR'], columns = ["MinVar","IVP","HRP"])

print(Perf_figures.to_latex(index=True)) # Latex table output

# Calculate performance figures

mean_MinVar = stats.mstats.gmean(np.array(portf_rets_90_2["MinVar"], dtype=float))-1
mean_IVP = stats.mstats.gmean(np.array(portf_rets_90_2["IVP"], dtype=float))-1
mean_HRP = stats.mstats.gmean(np.array(portf_rets_90_2["HRP"], dtype=float))-1

# Daily Standard deviation
std_MinVar = portf_rets_90["MinVar"].std()
std_IVP = portf_rets_90["IVP"].std()
std_HRP = portf_rets_90["HRP"].std()

# Sharpe ratios
SR_MinVar = mean_MinVar/std_MinVar
SR_IVP = mean_IVP/std_IVP
SR_HRP = mean_HRP/std_HRP

Perf_figures = pd.DataFrame([[mean_MinVar, mean_IVP,mean_HRP], [std_MinVar, \
                            std_IVP, std_HRP],[SR_MinVar,SR_IVP,SR_HRP]], \
    index =['Mean', 'Std', 'SR'], columns = ["MinVar","IVP","HRP"])

print(Perf_figures.to_latex(index=True)) # Latex table output



# In[ ]:

```

automatically created on 2019-11-18