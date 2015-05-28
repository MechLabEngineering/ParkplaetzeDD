
# coding: utf-8

# <img style="float: right;" alt="" width=250 src="http://pydata.org/berlin2015/static/img/PyDataBerlin-2015.png" />
# 
# # Analysing and predicting inner-city parking space occupancy
# 
# The city of Dresden has an excellent traffic monitoring and guiding system ([VAMOS](https://www.youtube.com/watch?v=zfWvjmlTXG4)), which also measures the occupancy rate of city parking spaces. The data is pushed to the [city's website](http://www.dresden.de/freie-parkplaetze/), from which it has been scraped by [Dresden's Open Data activists](http://codefor.de/projekte/2014-04-19-dd-freieparkplaetze.html) for the past year.
# 
# ![OpenData Scraping](http://mechlab-engineering.de/wordpress/wp-content/uploads/2015/03/DataFlow.png)

# In[49]:

from IPython.display import IFrame
IFrame('https://mopo24.de/nachrichten/park-chaos-truebt-die-heimelige-stimmung-2700', width='100%', height=450)


# This talk shows, how to analyse the data with Pandas and make predictions of future occupation with SciKit-Learn. Especially, which features are important for predicting the shopping behavior of citizens and tourists.
# 
# * Talk by [Paul Balzer](http://trustme.engineer), MechLab Engineering
# * PyData Berlin 2015, 29. - 30.05.2015, Betahaus Berlin

# ### Get a dump of the latest Data

# In[50]:

#!rm -rf parken_dump.csv
#!wget http://ubahn.draco.uberspace.de/opendata/dump/parken_dump.csv


# In[51]:

import pandas as pd
import numpy as np

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.dates as dates

import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('talk')


# [Here is a map for the parking spaces in Dresden](http://ubahn.draco.uberspace.de/opendata/ui/)
# 
# 
# And there is a dump of all the data! That is a line by line format, which is nice for logging/storing, but not for our purpose. So we have to pivot the data.

# In[52]:

data = pd.read_csv('parken_dump.csv', encoding='latin1')


# Calculate the percentage of occupation of the parking space

# In[53]:

data['Belegung'] = 100.0-data['free']/data['count']*100.0


# Let's create the dataframe

# In[54]:

ppDD = data.pivot(index='time', columns='name', values='Belegung')


# In[55]:

# Define index and names
ppDD.index = pd.DatetimeIndex(ppDD.index)
ppDD.index.name = 'Zeit'
ppDD.columns.name = 'Parkplatz'


# In[56]:

print('Daten von %s/%s bis %s/%s' % (ppDD.index[0].month, ppDD.index[0].year, ppDD.index[-1].month, ppDD.index[-1].year)) 


# In[57]:

# Wir nehmen nur ein gesamtes Jahr
ppDD = ppDD['2014-04-14':'2015-04-13']


# In[58]:

ppDD.head(3)


# In[59]:

ppDD.tail(3)


# ### Check for missing data
# 
# This function calculates the percentage of missing data points relative to the whole time. Some parking spaces are just open for special events or were just opened later in time. So the missing data % is not just error for sensors and stuff.

# In[60]:

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
    errordf.columns=['Fehlende Daten']
    errordf.sort(['Fehlende Daten'], inplace=True, ascending=False)
    
    # Plot it
    errordf.plot(kind='bar', figsize=(10,4))
    plt.ylabel(u'Keine Daten [% der Gesamtzeit]')
    plt.xlabel('Parkplatz')
    plt.annotate('Daten: dresden.de/freie-parkplaetze', xy=(0.95, 0.02), xycoords='figure fraction', ha='right')
    plt.annotate('CC-BY 2.0 Codefor.de/dresden', xy=(0.05, 0.02), xycoords='figure fraction', ha='left')
    plt.ylim(0, 100)
    return plt


# In[61]:

plt = checkformissingdata(ppDD)
plt.savefig('Fehlende-Daten.png', dpi=150, bbox_inches='tight')


# Argh, that's a lot!!

# ### Some interpolation or filling methods for the data

# In[62]:

# just some forward filling, means 8*15min = 2hours
#ppDD = ppDD.fillna(method='pad', limit=8) # Fill values forward

# Interpolation is possible, but the time betweet two datapoints might be a whole day!
#ppDD = ppDD.interpolate()

# Simply drop every row if a NaN is there, is a bad idea, because then you don't have very few data
#ppDD = ppDD.dropna()


# We do not use any of them, but we format the data in some ways

# In[63]:

# format the percent without digits
ppDD = ppDD.applymap(lambda x: float('%.0f' % x))

# and limit it between 0...100%
ppDD = ppDD.applymap(lambda x: min(max(x, 0.0), 100.0))


# In[64]:

ppDD.tail(3)


# ### Dump to Excel

# In[65]:

# dauert etwas länger
ppDD.to_excel('parken_dump.xlsx', sheet_name='Belegung')


# ### Heiligabend 2014

# In[66]:

ppDD['2014-12-21':'2014-12-29'][['Altmarkt']].plot(figsize=(10,5), title=u'Parkplatzbelegung in Dresden während der Weihnachtszeit 2014', ylim=(0, 130))
plt.ylabel('%')
plt.annotate('Daten: dresden.de/freie-parkplaetze', xy=(0.95, 0.02), xycoords='figure fraction', ha='right')
plt.annotate('CC-BY 2.0 Codefor.de/dresden', xy=(0.05, 0.02), xycoords='figure fraction', ha='left')
plt.axvline('2014-12-24 18:00', label='Heiligabend 18Uhr', alpha=0.2, c='k', lw=5)
plt.legend(ncol=2)
plt.savefig('Parkplatz-Dresden-Heiligabend-2014.png', dpi=72)


# ## Tagesgänge für jeden Parkplatz für jeden Monat

# In[67]:

weekend = ppDD['2014-06']['Altmarkt'].index[ppDD['2014-06']['Altmarkt'].index.weekday>4]


# In[68]:

ax = ppDD['2014-06']['Altmarkt'].plot(ylim=(0, 120), rot=0, figsize=(10,5))

ax.xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=(0,1,2,3,4,5,6),interval=2))
ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=(0),interval=1))
ax.xaxis.set_minor_formatter(dates.DateFormatter('%d.'))
ax.xaxis.set_major_formatter(dates.DateFormatter('\nSa/So'))
ax.xaxis.grid(True, which="minor")
ax.xaxis.grid(False, which="major")

for w in weekend[::10]:
    plt.axvline(w, c='k', alpha=0.08, zorder=-1)


# Now for all the Data! This may take a while!

# In[69]:

for pp in ppDD.columns: # alle Parkplätze
    
    print('Erstelle Plots: \'%s\'' % pp)
    
    if pp=='Centrum-Galerie':

        for y in range(2014, 2016): # zwischen 2014 und 2015
            for m in range(1, 13): # monatlich
                try:
                    d = '%s-%02d' % (y, m)
                    #print('Erstelle %s' % d)

                    ax = ppDD[d][pp].plot(figsize=(11,5), title=u'Parkplatzbelegung \'%s\' im %s %s' % (pp, ppDD[d].index[0].strftime('%b'), ppDD[d].index[0].strftime('%Y')), ylim=(0, 120), rot=0)

                    # Wochenende einzeichnen
                    weekend = ppDD[d][pp].index[ppDD[d][pp].index.weekday>4] # Samstag & Sonntag
                    for w in weekend[::13]:
                        plt.axvline(w, c='k', alpha=0.08, zorder=-1)

                    # Datumachse formatieren
                    ax.xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=(0,1,2,3,4,5,6),interval=1))
                    ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=(0),interval=1))
                    ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))
                    ax.xaxis.set_major_formatter(dates.DateFormatter('\nSa/So'))
                    ax.xaxis.grid(True, which="minor")
                    ax.xaxis.grid(False, which="major")

                    # Achsen beschriften
                    plt.ylabel('Belegung in %')
                    plt.annotate('Daten: dresden.de/freie-parkplaetze', xy=(0.98, 0.02), xycoords='figure fraction', ha='right')
                    plt.annotate('CC-BY 2.0 Codefor.de/dresden', xy=(0.02, 0.02), xycoords='figure fraction', ha='left')
                    outname = './Monatsbelegungen/%s-%s' % (pp.replace('/', ' '), d)
                    plt.savefig(outname, dpi=150)
                    plt.close()

                    #pass

                except:
                    # wenn die Daten nicht da sind, einfach überspringen
                    continue
                    
    else:
        print('Skipping...')
        continue


# In[70]:

def occupationbymonth(ppDD, fromtime, totime, which):
    ppDDmonthtag = ppDD.between_time(fromtime,totime).groupby(ppDD.between_time(fromtime,totime).index.month)
    ppDDmonthtagmean = ppDDmonthtag.aggregate(np.mean)

    ppDDmonthtagmean[which].plot(kind='bar', rot=0, ylim=(0, 100), width=0.7, figsize=(10,5))
    plt.xlabel('Monat')
    plt.ylabel('mittlere Belegung in %')

    plt.title(u'Parkplatz-Belegung zwischen %s und %sUhr' % (fromtime, totime))
    plt.annotate('Daten: dresden.de/freie-parkplaetze', xy=(0.98, 0.02), xycoords='figure fraction', ha='right')
    plt.annotate('CC-BY 2.0 Codefor.de/dresden', xy=(0.02, 0.02), xycoords='figure fraction', ha='left')
    plt.savefig('Mittlere-Belegung-mtl-%s-%sUhr.png' % (fromtime.replace(':','_'), totime.replace(':','_')), dpi=150, bbox_inches='tight')
    return plt


# ## Mittlere Belegung je Monat
# 
# ### 24h

# In[71]:

fromtime = '0:00'
totime = '23:59'
which = ['Altmarkt','Frauenkirche Neumarkt','An der Frauenkirche','Centrum-Galerie']
plt = occupationbymonth(ppDD, fromtime, totime, which)


# ### Tagsüber

# In[72]:

fromtime = '10:00'
totime = '18:00'
which = ['Altmarkt','Frauenkirche Neumarkt','An der Frauenkirche','Centrum-Galerie']
plt = occupationbymonth(ppDD, fromtime, totime, which)


# ### Nachts

# In[73]:

fromtime = '22:00'
totime = '6:00'
which = ['Altmarkt','Frauenkirche Neumarkt','An der Frauenkirche','Centrum-Galerie']
plt = occupationbymonth(ppDD, fromtime, totime, which)


# ## Mittlere Belegung nach Wochentag

# In[74]:

ppDDweekday = ppDD.groupby(ppDD.index.weekday)


# In[75]:

ppDDweekdaymean = ppDDweekday.aggregate(np.mean)


# In[76]:

ppDDweekdaymean.index = ['Montag','Dienstag','Mittwoch','Donnerstag','Freitag','Samstag','Sonntag']


# In[77]:

ppDDweekdaymean[['Altmarkt','Frauenkirche Neumarkt','An der Frauenkirche','Centrum-Galerie']].plot(kind='bar', rot=25, ylim=(0,100), figsize=(10,5))
plt.xlabel('Wochentag')
plt.ylabel('mittlere Belegung in %')

plt.annotate('Daten: dresden.de/freie-parkplaetze', xy=(0.98, 0.02), xycoords='figure fraction', ha='right')
plt.annotate('CC-BY 2.0 Codefor.de/dresden', xy=(0.02, 0.02), xycoords='figure fraction', ha='left')
plt.savefig('Mittlere-Belegung-Wochentag.png', dpi=150, bbox_inches='tight')


# ### Tagsüber

# In[78]:

def occupationbytime(df, fromtime, totime, which):
    ppDDweekdaytag = df.between_time(fromtime,totime).groupby(df.between_time(fromtime,totime).index.weekday)
    ppDDweekdaytagmean = ppDDweekdaytag.aggregate(np.mean)
    ppDDweekdaytagmean.index = ['Montag','Dienstag','Mittwoch','Donnerstag','Freitag','Samstag','Sonntag']
    
    ppDDweekdaytagmean[which].plot(kind='bar', figsize=(10,5))
    plt.xlabel('Wochentag')
    plt.ylabel('mittlere Belegung in %')
    plt.ylim(0,120)
    plt.title(u'Parkplatz-Belegung zwischen %s und %sUhr' % (fromtime, totime))
    plt.xticks(rotation=25)
    plt.annotate('Daten: dresden.de/freie-parkplaetze', xy=(0.98, 0.02), xycoords='figure fraction', ha='right')
    plt.annotate('CC-BY 2.0 Codefor.de/dresden', xy=(0.02, 0.02), xycoords='figure fraction', ha='left')
    plt.savefig('Mittlere-Belegung-%s-%sUhr.png' % (fromtime.replace(':','_'), totime.replace(':','_')), dpi=150, bbox_inches='tight')
    return plt


# In[79]:

plt = occupationbytime(ppDD, '10:00', '18:00', ['Altmarkt','Frauenkirche Neumarkt','An der Frauenkirche','Centrum-Galerie'])


# ### Nachts

# In[80]:

plt = occupationbytime(ppDD, '22:00', '6:00', ['Altmarkt','Frauenkirche Neumarkt','An der Frauenkirche','Centrum-Galerie'])


# In[ ]:




# ##Maximale Belegung an bestimmten Wochentagen je Kalenderwoche

# In[81]:

ppDDsamstag = ppDD[ppDD.index.weekday==5].groupby(ppDD[ppDD.index.weekday==5].index.week)
ppDDsamstagmax = ppDDsamstag.aggregate(np.max)


# In[82]:

ppDDsamstagmax['Centrum-Galerie'].plot(kind='bar', rot=0, ylim=(0, 100), width=0.7, figsize=(12,3))

plt.xlabel('Kalenderwoche')
plt.ylabel('maximale Belegung in %')

plt.title(u'Maximale Belegung des Parkhauses \'Centrum-Galerie\' an Samstagen')
plt.annotate('Daten: dresden.de/freie-parkplaetze', xy=(0.98, 0.02), xycoords='figure fraction', ha='right')
plt.annotate('CC-BY 2.0 Codefor.de/dresden', xy=(0.02, 0.02), xycoords='figure fraction', ha='left')
plt.savefig('Maximale-Belegung-Samstags-Centrum-Galerie.png', dpi=150, bbox_inches='tight')


# In[83]:

ppDDsonntag = ppDD[ppDD.index.weekday==6].groupby(ppDD[ppDD.index.weekday==6].index.week)
ppDDsonntagmax = ppDDsonntag.aggregate(np.max)


# In[84]:

ppDDsonntagmax['Centrum-Galerie'].plot(kind='bar', rot=0, ylim=(0, 100), width=0.7, figsize=(12,3))

plt.xlabel('Kalenderwoche')
plt.ylabel('maximale Belegung in %')

plt.title(u'Maximale Belegung des Parkhauses \'Centrum-Galerie\' an Sonntagen')
plt.annotate('Daten: dresden.de/freie-parkplaetze', xy=(0.98, 0.02), xycoords='figure fraction', ha='right')
plt.annotate('CC-BY 2.0 Codefor.de/dresden', xy=(0.02, 0.02), xycoords='figure fraction', ha='left')
plt.savefig('Maximale-Belegung-Sonntags-Centrum-Galerie.png', dpi=150, bbox_inches='tight')


# # Jahresnutzung nach KW und Wochentag

# In[85]:

weeks = ppDD['Centrum-Galerie'].index.week
weekdays = ppDD['Centrum-Galerie'].index.weekday


# In[86]:

jahresmean = ppDD['Centrum-Galerie'].groupby([weeks, weekdays]).aggregate(np.max).astype('int')


# In[87]:

jahresmean = jahresmean.unstack(level=0).fillna(0)


# In[88]:

#jahresmean.index = ['Montag','Dienstag','Mittwoch','Donnerstag','Freitag','Samstag','Sonntag']
jahresmean.index.name = 'Wochentag'
jahresmean.columns.name = 'KW'


# In[89]:

jahresmean


# In[90]:

Weekday, Week = np.mgrid[jahresmean.index[0]:jahresmean.index[-1]+2, jahresmean.columns[0]:jahresmean.columns[-1]+2]


# In[ ]:




# In[91]:

from matplotlib.colors import LinearSegmentedColormap
GrYeRe = LinearSegmentedColormap.from_list('name', ['green','green','green', 'yellow', 'red'])


# In[92]:

fig, ax = plt.subplots(figsize=(14, 3))
ax.set_aspect("equal")
pc = plt.pcolormesh(Week, Weekday, jahresmean.values, cmap=GrYeRe, edgecolor="w", vmin=0, vmax=100)

cbar=plt.colorbar(pc, shrink=0.5)
cbar.ax.set_ylabel(u'', rotation=270)
cbar.ax.set_xlabel(u'%')

plt.xlim(1, 53)
plt.ylim(7, 0)
ax.set_yticklabels(('Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag'), va='top')

xticks = range(1,53)
ax.set_xticks(xticks)
ax.set_xticklabels(xticks, ha='left')

#plt.grid()
plt.xlabel('Kalenderwoche');

pt = plt.title(u'Maximale Parkhausbelegung \'Centrum-Galerie\' Dresden nach Wochentag und Kalenderwoche\n                (Daten von %s/%s bis %s/%s)'                % (ppDD.index[0].month, ppDD.index[0].year, ppDD.index[-1].month, ppDD.index[-1].year), fontsize=11)

plt.annotate('Daten: dresden.de/freie-parkplaetze', xy=(0.96, 0.02), xycoords='figure fraction', ha='right')
plt.annotate('CC-BY 2.0 Codefor.de/dresden', xy=(0.02, 0.02), xycoords='figure fraction', ha='left')

plt.annotate('Heiligabend', xy=(52.4, 2.5), xytext=(45, 10), arrowprops={'arrowstyle': 'simple', 'facecolor': 'k'})
plt.annotate('verkaufsoffener Sonntag', xy=(17.5, 6.5), xytext=(10, 10), arrowprops={'arrowstyle': 'simple', 'facecolor': 'k'})


plt.tight_layout()

plt.savefig('Maximale-Belegung-Centrum-Galerie-KW-Jahr.png', dpi=150, bbox_inches='tight')


# In[ ]:




# In[ ]:



