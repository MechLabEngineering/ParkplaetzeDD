
# coding: utf-8

# ### Get a dump of the latest Data

# In[2]:

get_ipython().system(u'rm -rf parken_dump.csv')
get_ipython().system(u'wget http://ubahn.draco.uberspace.de/opendata/dump/parken_dump.csv')


# In[3]:

import pandas as pd
import numpy as np
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('talk')


# That is a line by line format, which is nice for logging/storing, but not for our purpose. So we have to pivot the data. Let's read in again, without index or something.

# In[4]:

data = pd.read_csv('parken_dump.csv', encoding='latin1')


# Calculate the percentage of occupation of the parking space

# In[5]:

data['Belegung'] = 100.0-data['free']/data['count']*100.0


# Let's create the dataframe

# In[6]:

ppDD = data.pivot(index='time', columns='name', values='Belegung')


# In[7]:

# Define index and names
ppDD.index = pd.DatetimeIndex(ppDD.index)
ppDD.index.name = 'Zeit'
ppDD.columns.name = 'Parkplatz'
print ppDD.index


# ### Check for missing data

# In[8]:

def checkformissingdata(df):
    '''
    input is a Pandas DataFrame
    output is a Matplotlib figure
    '''
    
    # create Dictionary
    error_rate={}
    for pp in ppDD.columns:
        # Error is percentage of NaN of whole dataset length
        error_rate[pp] = sum(pd.isnull(ppDD[pp]))*100.0/len(ppDD)
        
    # create Pandas Dataframe for easier plotting
    errordf = pd.DataFrame.from_dict(error_rate, orient='index')
    errordf.columns=['Fehlerhafte Daten']
    errordf.sort(['Fehlerhafte Daten'], inplace=True, ascending=False)
    
    # Plot it
    errordf.plot(kind='bar', figsize=(10,4))
    plt.ylabel(u'Keine Daten [in % der Gesamtzeit]')
    plt.xlabel('Parkplatz')
    plt.annotate('Daten: dresden.de/freie-parkplaetze', xy=(0.95, 0.02), xycoords='figure fraction', ha='right')
    plt.annotate('CC-BY 2.0 OpenKnowledge Foundation Dresden', xy=(0.05, 0.02), xycoords='figure fraction', ha='left')
    plt.ylim(0, 100)
    return plt


# In[9]:

plt = checkformissingdata(ppDD)
plt.savefig('Fehlerhafte-Daten.png', dpi=150, bbox_inches='tight')


# Argh, that's a lot!!

# ### Think we need some interpolation or filling of the data

# In[10]:

# just some forward filling, means 8*15min = 2hours
ppDD = ppDD.fillna(method='pad', limit=8) # Fill values forward

# Interpolation is possible, but the time betweet two datapoints might be a whole day!
#ppDD = ppDD.interpolate()

# Simply drop every row if a NaN is there, is a bad idea, because then you don't have very few data
#ppDD = ppDD.dropna()


# In[11]:

checkformissingdata(ppDD)


# Slightly better, but bad anyways for some parking spaces

# In[12]:

# format the percent with just 1 digit
ppDD = ppDD.applymap(lambda x: float('%.1f' % x))


# In[13]:

ppDD.tail(3)


# In[14]:

ppDD['2014-12-21':'2014-12-29'][['Altmarkt','Altmarkt - Galerie','Centrum-Galerie']].plot(figsize=(10,5), title=u'Parkplatzbelegung in Dresden während der Weihnachtszeit 2014', ylim=(0, 130))
plt.ylabel('%')
plt.annotate('Daten: dresden.de/freie-parkplaetze', xy=(0.95, 0.02), xycoords='figure fraction', ha='right')
plt.annotate('CC-BY 2.0 OpenKnowledge Foundation Dresden', xy=(0.05, 0.02), xycoords='figure fraction', ha='left')
plt.axvline('2014-12-24 18:00', label='Heiligabend 18Uhr', alpha=0.2, c='k', lw=5)
plt.legend(ncol=2)
plt.savefig('Parkplatz-Dresden-Heiligabend-2014.png', dpi=72)


# ## Tagesgänge für jeden Parkplatz für jeden Monat

# This takes a while!

# In[36]:

for pp in ppDD.columns:
    for y in range(2014, 2016):
        for m in range(1, 13):
            try:
                d = '%s-%02d' % (y, m)
                #print('Erstelle %s' % d)
                ppDD[d][pp].plot(figsize=(10,5), title=u'Parkplatzbelegung \'%s\'' % pp, ylim=(0, 130))
                plt.ylabel('%')
                plt.annotate('Daten: dresden.de/freie-parkplaetze', xy=(0.98, 0.02), xycoords='figure fraction', ha='right')
                plt.annotate('CC-BY 2.0 OpenKnowledge Foundation Dresden', xy=(0.02, 0.02), xycoords='figure fraction', ha='left')
                outname = './Monatsbelegungen/%s-%s' % (d, pp.replace('/', ' '))
                plt.savefig(outname, dpi=150, bbox_inches='tight')
                plt.close()
            except:
                continue


# ## Mean occupation by month

# In[38]:

ppDDmonth = ppDD.groupby(ppDD.index.month)


# In[39]:

ppDDmonthmean = ppDDmonth.aggregate(np.mean)


# In[40]:

ppDDmonthmean[['Altmarkt','Frauenkirche Neumarkt','Centrum-Galerie']].plot(kind='bar')
plt.xlabel('Monat')
plt.ylabel('mittlere Belegung in %')
plt.xticks(rotation=0)
plt.savefig('Mittlere-Belegung-mtl.png', dpi=150, bbox_inches='tight')


# ## Mean occupation by day in week

# In[41]:

ppDDweekday = ppDD.groupby(ppDD.index.weekday)


# In[42]:

ppDDweekdaymean = ppDDweekday.aggregate(np.mean)


# In[43]:

ppDDweekdaymean.index = ['Montag','Dienstag','Mittwoch','Donnerstag','Freitag','Samstag','Sonntag']


# In[44]:

ppDDweekdaymean[['Altmarkt','Frauenkirche Neumarkt','Centrum-Galerie']].plot(kind='bar')
plt.xlabel('Tag')
plt.ylabel('mittlere Belegung in %')
plt.xticks(rotation=25)
plt.savefig('Mittlere-Belegung-Tag.png', dpi=150, bbox_inches='tight')


# In[ ]:



