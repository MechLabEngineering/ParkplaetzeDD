#!/usr/bin/env python#!/usr/bin/python
# coding: utf-8

# # Evaluation of the predicted Occupation
# 
# we use the exported `.csv` of our `predictiveParking` and compare it with the true values.

# In[1]:


#!rm -rf parken_dump.csv
#!wget http://ubahn.draco.uberspace.de/opendata/dump/parken_dump.csv


# In[2]:


import pandas as pd


import matplotlib.pyplot as plt
import matplotlib.dates as dates

import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('talk')


# In[3]:


# function to plot timeseries with weekend
def plotbelegung(df, which, fromdate, todate):
    if fromdate=='':
        fromdate=df.index[0]
    if todate=='':
        todate=df.index[-1]
        
    weekend = df[fromdate:todate].index[df[fromdate:todate].index.weekday>4]
    ax = df[fromdate:todate][which].plot(figsize=(9,6), ylim=(0, 130), alpha=0.9, rot=0, title='Auslastung Parkhaus Centrum Galerie Dresden %s bis %s' % (fromdate, todate))
    
    if not df.index.freqstr: # Wenn DataFrame keine Frequenz hat
        ax.xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=(0,1,2,3,4,5,6),interval=2))
        ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=(0),interval=1))
        ax.xaxis.set_minor_formatter(dates.DateFormatter('%d.'))
        ax.xaxis.set_major_formatter(dates.DateFormatter('\nSa/So'))
        ax.xaxis.grid(True, which="minor")
        ax.xaxis.grid(False, which="major")
    
    plt.ylabel('%')
    
    for w in weekend[::10]:
        plt.axvline(w, c='k', alpha=0.08, zorder=-1)
        
    return plt


# In[36]:


data = pd.read_csv('dresdencentrumgalerie-2016.csv', names=['Datum','free'], index_col='Datum', parse_dates=True)


# In[37]:


data['Belegung'] = 100.0-(data.free/950.0*100.0)
data['Belegung'] = data['Belegung'].astype(int)
data.drop('free', axis=1, inplace=True)


# In[38]:


ppDD = data


# In[32]:


# Define index and names
ppDD.index = pd.DatetimeIndex(ppDD.index)
ppDD.index.name = 'Zeit'
ppDD.columns.name = 'Parkplatz'

print('Daten von %s/%s bis %s/%s' % (ppDD.index[0].month, ppDD.index[0].year, ppDD.index[-1].month, ppDD.index[-1].year)) 


# ### Machine Learning Model was trained with data until 2015-04-13, so we evaluate with the days after

# In[33]:


ppDD = ppDD['2015-04-14':]


# In[34]:


# format the percent without digits
ppDD = ppDD.applymap(lambda x: float('%.0f' % x))

# and limit it between 0...100%
ppDD = ppDD.applymap(lambda x: min(max(x, 0.0), 100.0))


# In[40]:


centrumGalerie = ppDD[['Belegung']].dropna()


# In[41]:


centrumGalerie.index = [idx.replace(second=0) for idx in centrumGalerie.index]


# In[42]:


centrumGalerie.head()


# ### Read the predicted data

# In[43]:


prediction = pd.read_csv('Centrum Galerie-Belegung-Vorhersage-2019-15min.csv', index_col=[0], skiprows=1, parse_dates=True, names=['Centrum-Galerie (Predicted)'])


# ### Join with the real data

# In[44]:


compare = centrumGalerie.join(prediction)
compare.dropna(how='any', inplace=True)


# In[45]:


compare.head(5)


# In[46]:


plotbelegung(compare, ['Centrum-Galerie','Centrum-Galerie (Predicted)'],'','')


# In[ ]:




